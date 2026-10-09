"""Durable trading state (SQLite, WAL).

Everything needed to resume after a crash or reboot is written *before*
the corresponding exchange action (write-ahead): an order row exists with
status ``pending`` before the transaction is signed, so a restart can always
reconcile what might have reached the exchange.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
  client_order_index INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL, market_id INTEGER NOT NULL, side INTEGER NOT NULL,
  qty REAL NOT NULL, price REAL NOT NULL, order_type TEXT NOT NULL, tif TEXT NOT NULL,
  reduce_only INTEGER NOT NULL, purpose TEXT NOT NULL, position_id TEXT,
  status TEXT NOT NULL, filled_qty REAL DEFAULT 0, avg_price REAL, fees REAL DEFAULT 0,
  tx_hash TEXT, exchange_order_index INTEGER, error TEXT, mode TEXT NOT NULL,
  t_decision INTEGER, t_created INTEGER, t_sent INTEGER, t_ack INTEGER, t_first_fill_print INTEGER, t_done INTEGER
);
CREATE INDEX IF NOT EXISTS orders_status ON orders(status);
CREATE TABLE IF NOT EXISTS positions (
  position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side INTEGER NOT NULL, qty REAL NOT NULL,
  entry_price REAL, entry_notional REAL, t_entry INTEGER, horizon_s REAL, exit_due_ns INTEGER,
  status TEXT NOT NULL, strategy TEXT, mu REAL, lcb REAL, exec_mode TEXT,
  exit_price REAL, t_exit INTEGER, pnl REAL, fees REAL DEFAULT 0, funding REAL DEFAULT 0, mode TEXT
);
CREATE INDEX IF NOT EXISTS positions_status ON positions(status);
CREATE TABLE IF NOT EXISTS fills (
  trade_id TEXT PRIMARY KEY, client_order_index INTEGER, t_ns INTEGER, price REAL, qty REAL,
  side INTEGER, fee REAL, liquidity TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, t_ns INTEGER, symbol TEXT, ok INTEGER, step INTEGER, reason TEXT,
  side INTEGER, mu REAL, lcb REAL, notional REAL, exec_mode TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS exec_latency (
  client_order_index INTEGER PRIMARY KEY, kind TEXT, t_decision INTEGER, t_sent INTEGER, t_ack INTEGER,
  t_fill_print INTEGER
);
CREATE TABLE IF NOT EXISTS equity (t_ns INTEGER PRIMARY KEY, equity REAL, available REAL, maint_req REAL, source TEXT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, t_ns INTEGER);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, t_ns INTEGER, kind TEXT, detail TEXT);
"""

OPEN_ORDER_STATES = ("pending", "sent", "open", "partially_filled", "unknown")


class StateStore:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # -- key/value -------------------------------------------------------------
    def put(self, key: str, value: Any) -> None:
        self.db.execute("INSERT INTO kv(key,value,t_ns) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                        " t_ns=excluded.t_ns", (key, json.dumps(value, default=str), time.time_ns()))

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def log_event(self, kind: str, detail: Any) -> None:
        self.db.execute("INSERT INTO events(t_ns,kind,detail) VALUES(?,?,?)",
                        (time.time_ns(), kind, json.dumps(detail, default=str)))

    # -- client order ids ----------------------------------------------------------
    def next_client_order_index(self) -> int:
        """Monotonic, crash-safe client order index (never reused)."""
        with self.db:
            cur = self.get("client_order_index", int(time.time() * 1000) % 10**12)
            nxt = int(cur) + 1
            self.put("client_order_index", nxt)
        return nxt

    # -- orders --------------------------------------------------------------------
    def insert_order(self, **o) -> None:
        cols = ",".join(o)
        self.db.execute(f"INSERT INTO orders({cols}) VALUES({','.join('?' * len(o))})", tuple(o.values()))

    def update_order(self, coi: int, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE orders SET {sets} WHERE client_order_index=?", (*fields.values(), coi))

    def order(self, coi: int) -> dict | None:
        r = self.db.execute("SELECT * FROM orders WHERE client_order_index=?", (coi,)).fetchone()
        return dict(r) if r else None

    def open_orders(self) -> list[dict]:
        q = f"SELECT * FROM orders WHERE status IN ({','.join('?' * len(OPEN_ORDER_STATES))})"
        return [dict(r) for r in self.db.execute(q, OPEN_ORDER_STATES)]

    # -- positions -------------------------------------------------------------------
    def upsert_position(self, **p) -> None:
        cols = ",".join(p)
        upd = ",".join(f"{k}=excluded.{k}" for k in p if k != "position_id")
        self.db.execute(f"INSERT INTO positions({cols}) VALUES({','.join('?' * len(p))}) "
                        f"ON CONFLICT(position_id) DO UPDATE SET {upd}", tuple(p.values()))

    def update_position(self, pid: str, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE positions SET {sets} WHERE position_id=?", (*fields.values(), pid))

    def positions(self, statuses=("opening", "open", "closing")) -> list[dict]:
        q = f"SELECT * FROM positions WHERE status IN ({','.join('?' * len(statuses))})"
        return [dict(r) for r in self.db.execute(q, statuses)]

    def closed_positions(self, limit: int = 1000) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM positions WHERE status='closed' ORDER BY t_exit DESC LIMIT ?", (limit,))]

    # -- fills / decisions / latency / equity -----------------------------------------
    def insert_fill(self, trade_id: str, coi: int, t_ns: int, price: float, qty: float, side: int, fee: float,
                    liquidity: str) -> bool:
        cur = self.db.execute("INSERT OR IGNORE INTO fills VALUES(?,?,?,?,?,?,?,?)",
                              (trade_id, coi, t_ns, price, qty, side, fee, liquidity))
        return cur.rowcount == 1

    def record_decision(self, d: dict) -> None:
        self.db.execute(
            "INSERT INTO decisions(t_ns,symbol,ok,step,reason,side,mu,lcb,notional,exec_mode,payload) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (d["t_ns"], d["symbol"], int(d["ok"]), d["step"], d["reason"], d.get("side", 0), d.get("mu"), d.get("lcb"),
             d.get("notional", 0.0), d.get("exec_mode", ""), json.dumps(d.get("extra", {}), default=str)))

    def record_latency(self, coi: int, kind: str, **t) -> None:
        cols = ["client_order_index", "kind", *t]
        upd = ",".join(f"{k}=COALESCE(excluded.{k},{k})" for k in t)
        self.db.execute(f"INSERT INTO exec_latency({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
                        f"ON CONFLICT(client_order_index) DO UPDATE SET {upd}", (coi, kind, *t.values()))

    def latency_records(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM exec_latency WHERE t_fill_print IS NOT NULL")]

    def record_equity(self, t_ns: int, equity: float, available: float, maint_req: float, source: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity VALUES(?,?,?,?,?)", (t_ns, equity, available, maint_req, source))

    def last_equity(self) -> dict | None:
        r = self.db.execute("SELECT * FROM equity ORDER BY t_ns DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def high_water(self) -> float:
        r = self.db.execute("SELECT MAX(equity) AS hw FROM equity").fetchone()
        return float(r["hw"]) if r and r["hw"] is not None else 0.0
