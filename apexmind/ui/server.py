"""Local control panel: API credentials, service start/stop, trader mode,
kill switch and the exchange integration test.

It runs as root (it writes /etc/apexmind and calls systemctl) and binds to
127.0.0.1, so it is opened from a browser on the VPS itself. Every request needs
the token in /etc/apexmind/ui_token. The stored API key is write-only here: it
is never sent back to the browser. There is no withdrawal or transfer function.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from apexmind.config import load_config

log = logging.getLogger(__name__)

SERVICES = {"collector": "apexmind-collector", "lab": "apexmind-lab", "trader": "apexmind-trader"}
MODE_ARGS = {"paper": "--mode paper", "live": "--mode live --i-understand-live-risk"}
# Lighter API private keys are 40 bytes (80 hex chars). Ethereum wallet keys
# are 32 bytes (64 hex), so a wallet key is rejected rather than stored.
API_KEY_RE = re.compile(r"^(0x)?[0-9a-fA-F]{80}$")
WALLET_KEY_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")
COOKIE = "apexmind_ui"

Runner = Callable[[list[str]], tuple[int, str]]


def _run(argv: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, repr(e)
    return p.returncode, (p.stdout + p.stderr).strip()


class PanelError(ValueError):
    pass


class Panel:
    def __init__(self, config_path: str, *, key_path: str | None = None,
                 token_path: str = "/etc/apexmind/ui_token",
                 dropin_dir: str = "/etc/systemd/system/apexmind-trader.service.d",
                 workdir: str = "/var/lib/apexmind", apexmind_bin: str = "/opt/apexmind/venv/bin/apexmind",
                 runner: Runner = _run):
        self.config_path = Path(config_path)
        cfg = load_config(self.config_path)
        self.key_path = Path(key_path or cfg.live.secrets_file)
        self.token_path = Path(token_path)
        self.dropin = Path(dropin_dir) / "mode.conf"
        self.workdir = Path(workdir)
        self.bin = apexmind_bin
        self.run = runner
        self._itest: subprocess.Popen | None = None
        self.itest_log = self.workdir / "state" / "ui_integration_test.log"

    # ---- token -------------------------------------------------------------------
    def token(self) -> str:
        if not self.token_path.exists():
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            _write_private(self.token_path, secrets.token_urlsafe(18) + "\n")
        return self.token_path.read_text().strip()

    # ---- state -------------------------------------------------------------------
    def cfg(self):
        return load_config(self.config_path)

    def mode(self) -> str:
        if self.dropin.exists() and "--mode live" in self.dropin.read_text():
            return "live"
        return "paper"

    def kill_path(self) -> Path:
        p = Path(self.cfg().live.kill_file)
        return p if p.is_absolute() else self.workdir / p

    def champion(self) -> str | None:
        from apexmind.cli import champion_id

        d = Path(self.cfg().lab.registry_dir)
        return champion_id(d if d.is_absolute() else self.workdir / d)

    def status(self) -> dict:
        from apexmind.cli import collect_status

        cfg = self.cfg()
        services = {}
        for name, unit in SERVICES.items():
            _, active = self.run(["systemctl", "is-active", unit])
            _, enabled = self.run(["systemctl", "is-enabled", unit])
            services[name] = {"active": active.splitlines()[-1] if active else "unknown",
                              "enabled": enabled.splitlines()[-1] if enabled else "unknown"}
        key = None
        if self.key_path.exists():
            key = {"set": True, "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(self.key_path.stat().st_mtime))}
        cwd = os.getcwd()
        try:
            os.chdir(self.workdir)  # the config's state/runs paths are relative to the services' workdir
            health = collect_status(cfg)
        except Exception as e:
            health = {"error": repr(e)}
        finally:
            os.chdir(cwd)
        health.pop("integration", None)
        itest = None
        rp = Path(cfg.live.integration_report)
        rp = rp if rp.is_absolute() else self.workdir / rp
        if rp.exists():
            try:
                itest = {"passed": json.loads(rp.read_text()).get("passed"),
                         "age_s": round(time.time() - rp.stat().st_mtime)}
            except (OSError, ValueError):
                itest = None
        return {
            "services": services,
            "credentials": {"account_index": cfg.lighter.account_index, "api_key_index": cfg.lighter.api_key_index,
                            "key": key or {"set": False}},
            "mode": self.mode(),
            "kill": self.kill_path().exists(),
            "integration": {"running": self._itest is not None and self._itest.poll() is None,
                            **(itest or {"passed": None, "age_s": None})},
            "health": {k: _summary(k, v) for k, v in health.items()},
        }

    # ---- actions -----------------------------------------------------------------
    def set_credentials(self, account_index, api_key_index, private_key: str | None) -> str:
        acct, key_idx = _int(account_index, "account index", 0), _int(api_key_index, "API key index", 0, 254)
        if key_idx < 4:
            raise PanelError("API key indexes 0-3 are reserved by Lighter's apps; use 4-254")
        msg = []
        if private_key:
            k = private_key.strip()
            if WALLET_KEY_RE.match(k):
                raise PanelError("This looks like an Ethereum WALLET private key (64 hex characters). Never put a "
                                 "wallet key on the server. Use the Lighter API private key (80 hex characters) "
                                 "from Lighter > Tools > API Keys.")
            if not API_KEY_RE.match(k):
                raise PanelError("The Lighter API private key must be 80 hex characters")
            _write_private(self.key_path, k.removeprefix("0x") + "\n")
            msg.append("API key saved")
        _set_lighter_indexes(self.config_path, acct, key_idx)
        msg.append(f"account {acct}, key index {key_idx} saved")
        if self._active("trader"):
            msg.append("restart the trader to use the new credentials")
        return "; ".join(msg)

    def service(self, name: str, action: str) -> str:
        unit = SERVICES.get(name)
        if unit is None or action not in ("start", "stop", "restart"):
            raise PanelError("unknown service or action")
        if name == "trader" and action != "stop" and self.champion() is None:
            raise PanelError("No strategy has been promoted yet, so the trader has nothing to trade and would "
                             "refuse to start. The lab needs about 60 hours of recorded data for its first run; "
                             "the champion appears under Status when one passes.")
        argv = {"start": ["systemctl", "enable", "--now", unit], "stop": ["systemctl", "disable", "--now", unit],
                "restart": ["systemctl", "restart", unit]}[action]
        rc, out = self.run(argv)
        if rc != 0:
            raise PanelError(f"{' '.join(argv)} failed: {out}")
        return f"{name}: {action} done"

    def set_mode(self, mode: str, confirm: str = "") -> str:
        if mode not in MODE_ARGS:
            raise PanelError("mode must be paper or live")
        if mode == "live" and confirm != "LIVE":
            raise PanelError("type LIVE to confirm live trading")
        self.dropin.parent.mkdir(parents=True, exist_ok=True)
        self.dropin.write_text(f"[Service]\nEnvironment=APEXMIND_TRADE_ARGS={MODE_ARGS[mode]}\n")
        self.run(["systemctl", "daemon-reload"])
        msg = f"trader mode set to {mode}"
        if self._active("trader"):
            rc, out = self.run(["systemctl", "restart", SERVICES["trader"]])
            msg += "; trader restarted" if rc == 0 else f"; restart failed: {out}"
        if mode == "live":
            msg += ". The trader still refuses live unless a live champion and a passing integration test " \
                   "under 24 h old exist; check its log."
        return msg

    def set_kill(self, on: bool) -> str:
        p = self.kill_path()
        if on:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(time.strftime("%Y-%m-%dT%H:%M:%S killed from control panel\n"))
            return "KILL switch ON: the trader halts entries and flattens positions"
        p.unlink(missing_ok=True)
        return "KILL switch OFF: trading may resume"

    def start_integration_test(self, place_orders: bool) -> str:
        if self._itest is not None and self._itest.poll() is None:
            raise PanelError("an integration test is already running")
        if not self.key_path.exists():
            raise PanelError("save the API key first")
        argv = ["systemd-run", "--quiet", "--collect", "--wait", "--pipe", "--uid=apexmind", "--gid=apexmind",
                "-p", f"LoadCredential=lighter_api_key:{self.key_path}", "-p", f"WorkingDirectory={self.workdir}",
                self.bin, "--config", str(self.config_path), "integration-test"]
        if place_orders:
            argv += ["--place-test-order", "--latency-samples", "30"]
        self.itest_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.itest_log, "w") as fh:
            fh.write(f"$ apexmind integration-test{' --place-test-order --latency-samples 30' if place_orders else ''}\n")
            fh.flush()
            self._itest = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        return "integration test started" + (" (places small real orders)" if place_orders else " (no orders)")

    def integration_output(self) -> str:
        return self.itest_log.read_text()[-20000:] if self.itest_log.exists() else ""

    def logs(self, name: str) -> str:
        unit = SERVICES.get(name)
        if unit is None:
            raise PanelError("unknown service")
        _, out = self.run(["journalctl", "-u", unit, "-n", "120", "--no-pager", "-o", "short-iso"])
        return out

    def _active(self, name: str) -> bool:
        _, out = self.run(["systemctl", "is-active", SERVICES[name]])
        return out.strip().endswith("active") and not out.strip().endswith("inactive")


def _summary(name: str, v):
    if not isinstance(v, dict):
        return v
    if name == "collector":
        q = v.get("quality") or {}
        return {"age_s": v.get("age_s"), "streams": len(q), "gaps": sum(s.get("gaps", 0) for s in q.values()),
                "hours_since_restart": max((s.get("span_hours", 0) for s in q.values()), default=0)}
    if name == "lab" and v.get("status") == "waiting_for_data":
        return {"status": "waiting for data", "recorded_hours": v.get("recorded_hours"),
                "first_run_at_hours": v.get("first_run_at_hours"), "full_test_hours": v.get("full_protocol_hours")}
    if name == "lab" and v.get("status") == "error":
        return {"status": "error (see lab log)"}
    if name == "lab":
        res = v.get("results") or {}
        return {"age_s": v.get("age_s"), "last_run": time.strftime("%m-%d %H:%M", time.localtime(v.get("t", 0))),
                "promoted": [r.get("promoted") for r in res.values() if isinstance(r, dict) and r.get("promoted")]}
    return {k: v[k] for k in list(v)[:6]}


def _int(v, what: str, lo: int, hi: int | None = None) -> int:
    try:
        i = int(str(v).strip())
    except ValueError:
        raise PanelError(f"{what} must be a whole number") from None
    if i < lo or (hi is not None and i > hi):
        raise PanelError(f"{what} out of range")
    return i


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _set_lighter_indexes(path: Path, account_index: int, api_key_index: int) -> None:
    """Set lighter.account_index/api_key_index in place, keeping comments."""
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if re.match(r"^lighter:\s*(#.*)?$", ln)), None)
    if start is None:
        lines += ["lighter:\n"]
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^\S", lines[i])), len(lines))
    for key, val in (("account_index", account_index), ("api_key_index", api_key_index)):
        pat = re.compile(rf"^(\s+{key}:)\s*[^#\n]*?(\s*#.*)?$")
        for i in range(start + 1, end):
            m = pat.match(lines[i].rstrip("\n"))
            if m:
                lines[i] = f"{m.group(1)} {val}{m.group(2) or ''}\n"
                break
        else:
            lines.insert(start + 1, f"  {key}: {val}\n")
            end += 1
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(lines))
    st = path.stat()
    os.chmod(tmp, st.st_mode & 0o777)
    try:
        os.chown(tmp, st.st_uid, st.st_gid)
    except PermissionError:
        pass
    os.replace(tmp, path)


def make_handler(panel: Panel, allowed_hosts: set[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "apexmind-ui"

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj, headers=None):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json", headers)

        def _host_ok(self) -> bool:
            # Rejects DNS-rebinding: the page is only valid under its local address.
            return self.headers.get("Host", "") in allowed_hosts

        def _authed(self) -> bool:
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE and hmac.compare_digest(v, panel.token()):
                    return True
            return False

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 10000:
                raise PanelError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden host", "text/plain")
            if self.path == "/":
                page = PAGE if self._authed() else LOGIN
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if not self._authed():
                return self._json(401, {"error": "login required"})
            try:
                if self.path == "/api/status":
                    return self._json(200, panel.status())
                if self.path == "/api/itest":
                    return self._json(200, {"output": panel.integration_output(),
                                            "running": panel.status()["integration"]["running"]})
                if self.path.startswith("/api/logs/"):
                    return self._json(200, {"output": panel.logs(self.path.rsplit("/", 1)[1])})
            except PanelError as e:
                return self._json(400, {"error": str(e)})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"forbidden host", "text/plain")
            if self.path == "/login":
                n = int(self.headers.get("Content-Length") or 0)
                form = dict(p.partition("=")[::2] for p in self.rfile.read(min(n, 2000)).decode().split("&") if p)
                from urllib.parse import unquote_plus
                if hmac.compare_digest(unquote_plus(form.get("token", "")), panel.token()):
                    return self._send(303, b"", "text/plain", {
                        "Location": "/",
                        "Set-Cookie": f"{COOKIE}={panel.token()}; HttpOnly; SameSite=Strict; Path=/"})
                time.sleep(1.0)
                return self._send(200, LOGIN.replace("<!--err-->", "<p class=err>Wrong token</p>").encode(),
                                  "text/html; charset=utf-8")
            # Custom header + JSON body: a cross-site page cannot send this without a CORS preflight.
            if not self._authed() or self.headers.get("X-Apexmind") != "1":
                return self._json(401, {"error": "login required"})
            try:
                b = self._body()
                if self.path == "/api/credentials":
                    msg = panel.set_credentials(b.get("account_index"), b.get("api_key_index"), b.get("private_key"))
                elif self.path == "/api/service":
                    msg = panel.service(str(b.get("name")), str(b.get("action")))
                elif self.path == "/api/mode":
                    msg = panel.set_mode(str(b.get("mode")), str(b.get("confirm", "")))
                elif self.path == "/api/kill":
                    msg = panel.set_kill(bool(b.get("on")))
                elif self.path == "/api/itest":
                    msg = panel.start_integration_test(bool(b.get("place_orders")))
                else:
                    return self._json(404, {"error": "not found"})
            except (PanelError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, {"message": msg})

    return Handler


def serve(config_path: str | None, bind: str = "127.0.0.1", port: int = 18787, **panel_kw) -> int:
    if not config_path:
        print("--config is required (e.g. /etc/apexmind/config.yaml)", file=sys.stderr)
        return 2
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        log.warning("not running as root: saving credentials and starting services will fail")
    panel = Panel(config_path, **panel_kw)
    panel.token()
    hosts = {f"{h}:{port}" for h in ("127.0.0.1", "localhost", bind)}
    httpd = ThreadingHTTPServer((bind, port), make_handler(panel, hosts))
    log.info("control panel on http://%s:%d (token in %s)", bind, port, panel.token_path)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


LOGIN = """<!doctype html><html><head><meta charset=utf-8><title>APEX MIND</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;display:flex;justify-content:center;
padding-top:12vh}form{background:#181b22;padding:28px;border-radius:10px;width:340px}input{width:100%;padding:9px;
margin:10px 0;background:#0f1115;color:#e6e6e6;border:1px solid #333;border-radius:6px;box-sizing:border-box}
button{width:100%;padding:10px;background:#2f6feb;color:#fff;border:0;border-radius:6px;cursor:pointer}
.err{color:#ff6b6b}small{color:#999}</style></head><body><form method=post action=/login>
<h2>APEX MIND control panel</h2><!--err--><label>Access token</label>
<input name=token type=password autofocus autocomplete=off>
<small>Show it on the VPS with: <code>sudo cat /etc/apexmind/ui_token</code></small><br><br>
<button>Open</button></form></body></html>"""

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>APEX MIND</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:20px;max-width:1000px;
margin:auto}h1{font-size:20px}h2{font-size:15px;margin:0 0 12px;color:#9ab}
.card{background:#181b22;border-radius:10px;padding:16px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
label{display:block;font-size:13px;color:#aaa;margin-top:8px}
input,select{width:100%;padding:8px;background:#0f1115;color:#e6e6e6;border:1px solid #333;border-radius:6px;
box-sizing:border-box}button{padding:8px 14px;border:0;border-radius:6px;cursor:pointer;color:#fff;
background:#2f6feb;margin:4px 4px 0 0}button.stop{background:#a33}button.warn{background:#b7791f}
button.grey{background:#444}table{width:100%;border-collapse:collapse}td{padding:6px 4px;border-bottom:1px solid #252a33}
.on{color:#3fb950}.off{color:#ff7b72}.muted{color:#888;font-size:12px}
pre{background:#0b0d10;padding:10px;border-radius:6px;max-height:340px;overflow:auto;font-size:12px;white-space:pre-wrap}
#msg{position:sticky;top:0;padding:10px;border-radius:6px;display:none;margin-bottom:10px}
.ok{background:#12361f}.bad{background:#4a1515}
</style></head><body>
<h1>APEX MIND control panel</h1>
<div id=msg></div>

<div class=card><h2>Status</h2><div class=grid id=health></div></div>

<div class=card><h2>Services</h2><table id=svc></table>
<p class=muted>Stopping the trader closes its open positions. Start/Stop also sets whether the service starts after a reboot.</p></div>

<div class=card><h2>Lighter API credentials</h2>
<div class=grid>
<div><label>Account index</label><input id=acct inputmode=numeric></div>
<div><label>API key index (4-254)</label><input id=kidx inputmode=numeric></div>
</div>
<label>API private key <span class=muted id=keystate></span></label>
<input id=pkey type=password autocomplete=off placeholder="leave empty to keep the saved key">
<p class=muted>Use only the API key from Lighter &rsaquo; Tools &rsaquo; API Keys. Never your wallet seed phrase or wallet
private key. The saved key is never shown again.</p>
<button onclick=saveCreds()>Save</button></div>

<div class=card><h2>Trader mode</h2>
<div>Current: <b id=mode></b></div>
<button class=grey onclick="setMode('paper')">Paper (simulated)</button>
<button class=warn onclick="setMode('live')">Live (real money)</button>
<p class=muted>Live still requires a promoted live champion and a passing integration test under 24 h old; otherwise
the trader refuses to start.</p></div>

<div class=card><h2>Kill switch</h2>
<div>State: <b id=kill></b></div>
<button class=stop onclick="kill(true)">KILL: halt and flatten</button>
<button class=grey onclick="kill(false)">Release</button></div>

<div class=card><h2>Exchange integration test</h2>
<div id=itstate class=muted></div>
<button class=grey onclick="itest(false)">Check connection and key (no orders)</button>
<button class=warn onclick="itest(true)">Full test (small real orders, ~cents)</button>
<pre id=itout></pre></div>

<div class=card><h2>Logs</h2>
<select id=logunit onchange=loadLogs()><option>trader</option><option>collector</option><option>lab</option></select>
<button class=grey onclick=loadLogs()>Refresh</button><pre id=logs></pre></div>

<script>
const $=id=>document.getElementById(id);
function show(t,ok){const m=$('msg');m.textContent=t;m.className=ok?'ok':'bad';m.style.display='block';
 clearTimeout(window._mt);window._mt=setTimeout(()=>m.style.display='none',9000)}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json',
 'X-Apexmind':'1'},body:JSON.stringify(body)});const j=await r.json().catch(()=>({error:'bad response'}));
 if(r.status==401){location.reload();return}show(j.message||j.error,r.ok);refresh();return r.ok}
function cell(tr,text,cls){const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;
 tr.appendChild(td);return td}
let filled=false;
async function refresh(){const r=await fetch('/api/status');if(r.status==401){location.reload();return}
 const s=await r.json();const t=$('svc');t.replaceChildren();
 for(const [n,v] of Object.entries(s.services)){const tr=document.createElement('tr');cell(tr,n);
  cell(tr,v.active,v.active=='active'?'on':'off');cell(tr,'autostart: '+v.enabled,'muted');
  const td=cell(tr,'');for(const a of ['start','stop','restart']){const b=document.createElement('button');
   b.textContent=a;b.className=a=='stop'?'stop':(a=='restart'?'grey':'');
   b.onclick=()=>{if(a=='stop'&&n=='trader'&&!confirm('Stop the trader? Open positions are closed.'))return;
    post('/api/service',{name:n,action:a})};td.appendChild(b)}t.appendChild(tr)}
 const c=s.credentials;if(!filled){$('acct').value=c.account_index>=0?c.account_index:'';
  $('kidx').value=c.api_key_index>=0?c.api_key_index:'';filled=true}
 $('keystate').textContent=c.key.set?'(saved '+c.key.updated+')':'(not set)';
 $('mode').textContent=s.mode.toUpperCase();$('mode').className=s.mode=='live'?'off':'on';
 $('kill').textContent=s.kill?'ON (trading halted)':'off';$('kill').className=s.kill?'off':'on';
 const it=s.integration;$('itstate').textContent=it.running?'Running...':(it.passed==null?'Never run':
  ('Last result: '+(it.passed?'PASSED':'FAILED')+', '+Math.round(it.age_s/3600*10)/10+' h ago'));
 const h=$('health');h.replaceChildren();for(const [k,v] of Object.entries(s.health)){const d=document.createElement('div');
  const b=document.createElement('b');b.textContent=k;const p=document.createElement('pre');
  p.textContent=typeof v=='object'&&v?JSON.stringify(v,null,1):String(v);d.append(b,p);h.appendChild(d)}
 if(it.running)loadItest()}
async function saveCreds(){const k=$('pkey').value.trim();
 if(await post('/api/credentials',{account_index:$('acct').value,api_key_index:$('kidx').value,private_key:k}))$('pkey').value=''}
function setMode(m){let confirmText='';if(m=='live'){confirmText=prompt('Live trading uses real money. Type LIVE to confirm.');
 if(confirmText===null)return}post('/api/mode',{mode:m,confirm:confirmText})}
function kill(on){if(on&&!confirm('Halt trading and close all positions?'))return;post('/api/kill',{on})}
async function itest(full){if(full&&!confirm('Places a post-only order and 30 tiny round-trip orders (costs cents). Continue?'))return;
 if(await post('/api/itest',{place_orders:full}))setTimeout(loadItest,1500)}
async function loadItest(){const r=await fetch('/api/itest');const j=await r.json();$('itout').textContent=j.output||''}
async function loadLogs(){const r=await fetch('/api/logs/'+$('logunit').value);const j=await r.json();
 $('logs').textContent=j.output||j.error||'';$('logs').scrollTop=1e9}
refresh();loadItest();loadLogs();setInterval(refresh,5000);
</script></body></html>"""
