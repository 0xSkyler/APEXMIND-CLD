"""Command-line entry point: ``apexmind <command> [options]``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shlex
import sys
import time
from pathlib import Path

from apexmind.config import load_config


def _ts(s: str | None) -> int | None:
    if not s:
        return None
    import pandas as pd

    return int(pd.Timestamp(s, tz="UTC").value)


def _span(data_dir: str, cfg, start: str | None, end: str | None, hours: float | None) -> tuple[int, int]:
    from apexmind.collector.recorder import data_span

    sp = data_span(data_dir, ["lighter", cfg.reference.venue])
    if sp is None:
        raise SystemExit(f"no recorded market data under {data_dir}")
    s, e = _ts(start) or sp[0], _ts(end) or sp[1] + 1
    if hours:
        s = max(s, e - int(hours * 3.6e12))
    return s, e


def cmd_collect(a, cfg) -> int:
    from apexmind.collector.service import run_collector

    run_collector(cfg)
    return 0


def cmd_synthetic(a, cfg) -> int:
    from apexmind.data.synthetic import SymbolSpec, SyntheticSpec, generate

    spec = SyntheticSpec(hours=a.hours, lag_s=a.lag, seed=a.seed,
                         symbols=[SymbolSpec(), SymbolSpec(symbol="SOL", market_id=1, price=150.0, tick=0.01)])
    truth = generate(spec, a.out)
    print(json.dumps({"out": a.out, "messages": truth["messages"], "lag_s": a.lag,
                      "warning": "SYNTHETIC data for methodology validation only"}))
    return 0


def cmd_research(a, cfg) -> int:
    from apexmind.data.dataset import build_dataset
    from apexmind.report.proof import write_report
    from apexmind.research.pipeline import run_research

    start, end = _span(a.data, cfg, a.start, a.end, a.hours)
    df, meta = build_dataset(cfg, a.data, start, end, cache_dir=str(Path(cfg.research.runs_dir) / "cache"))
    run_dir = Path(a.run_dir or Path(cfg.research.runs_dir) / f"research-{time.strftime('%Y%m%dT%H%M%S')}")
    res = run_research(cfg, df, meta, run_dir, a.models.split(",") if a.models else None)
    cmd = "apexmind " + " ".join(shlex.quote(x) for x in sys.argv[1:])
    out = write_report(run_dir, cmd)
    print(json.dumps({"run_dir": str(run_dir), "report": str(out), "champion": res["champion"].get("selected"),
                      "verdict": res["champion"]["verdict"], "data_kind": meta.data_kind}, default=str))
    return 0


def cmd_report(a, cfg) -> int:
    from apexmind.report.proof import write_report

    print(write_report(a.run_dir))
    return 0


def cmd_lab(a, cfg) -> int:
    from apexmind.lab.laboratory import AlphaLab, apply_resource_limits

    lab = AlphaLab(cfg, a.data, a.purpose)
    if a.once:
        apply_resource_limits(cfg.lab.nice, cfg.lab.max_memory_gb, cfg.lab.max_threads)
        out = lab.run_cycle(a.hours)
        print(json.dumps({k: v for k, v in out.items() if k != "drift_psi"}, default=str)[:4000])
    else:
        lab.run_forever(a.hours)
    return 0


def cmd_trade(a, cfg) -> int:
    from apexmind.live.trader import PreflightError, run_trader

    mode = a.mode or cfg.live.mode
    if mode == "live" and not a.i_understand_live_risk:
        print("refusing live mode without --i-understand-live-risk", file=sys.stderr)
        return 2
    try:
        asyncio.run(run_trader(cfg, mode))
    except PreflightError as e:
        print(str(e), file=sys.stderr)
        return 3
    return 0


def cmd_integration(a, cfg) -> int:
    from apexmind.live.integration import main_integration

    return main_integration(cfg, a.place_test_order, a.latency_samples)


def collect_status(cfg) -> dict:
    from apexmind.lab.registry import Registry

    out = {}
    for name, p in (("trader", cfg.live.heartbeat_path), ("collector", cfg.collector.health_path),
                    ("lab", str(Path(cfg.research.runs_dir) / "lab_status.json")),
                    ("integration", cfg.live.integration_report)):
        path = Path(p)
        if path.exists():
            d = json.loads(path.read_text())
            out[name] = {"age_s": round(time.time() - path.stat().st_mtime, 1),
                         **({k: d[k] for k in list(d)[:12]} if isinstance(d, dict) else {})}
        else:
            out[name] = "missing"
    reg = Registry(cfg.lab.registry_dir)
    out["champion"] = reg.champion_id()
    return out


def cmd_status(a, cfg) -> int:
    print(json.dumps(collect_status(cfg), indent=1, default=str))
    return 0


def cmd_ui(a, cfg) -> int:
    from apexmind.ui.server import serve

    return serve(a.config, a.bind, a.port)


def cmd_validate(a, cfg) -> int:
    """Positive/negative synthetic controls through the full pipeline."""
    from apexmind.data.dataset import build_dataset
    from apexmind.data.synthetic import SymbolSpec, SyntheticSpec, generate
    from apexmind.report.proof import write_report
    from apexmind.research.pipeline import run_research

    out = Path(a.out)
    cfg.labels.latency_source = "prior"
    cfg.research.min_train_hours = a.hours / 3
    cfg.research.test_hours = a.hours / 6
    cfg.research.n_folds = 3
    cfg.research.block_minutes = 10.0
    summary = {}
    for tag, lag in (("positive_control", a.lag), ("negative_control", 0.0)):
        spec = SyntheticSpec(hours=a.hours, lag_s=lag, seed=a.seed,
                             symbols=[SymbolSpec(), SymbolSpec(symbol="SOL", market_id=1, price=150.0, tick=0.01)])
        raw = out / tag / "raw"
        if not raw.exists():
            generate(spec, str(raw))
        df, meta = build_dataset(cfg, str(raw), spec.start_ns + 5 * 60 * 10**9,
                                 spec.start_ns + int(a.hours * 3.6e12), cache_dir=str(out / "cache"))
        res = run_research(cfg, df, meta, out / tag / "run")
        write_report(out / tag / "run", f"apexmind validate-methodology --out {a.out} --hours {a.hours} "
                                        f"--lag {a.lag} --seed {a.seed}")
        summary[tag] = {"planted_lag_s": lag, "selected": res["champion"].get("selected"),
                        "verdict": res["champion"]["verdict"],
                        "beats_baselines": res["champion"].get("beats_all_baselines_after_costs")}
    ok = summary["positive_control"]["selected"] is not None and summary["negative_control"]["selected"] is None
    summary["methodology_ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="apexmind", description=__doc__)
    p.add_argument("--config", default=None, help="YAML config (defaults built in)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect", help="run the synchronized market-data collector")
    s = sub.add_parser("synthetic", help="write a SYNTHETIC recording (methodology validation only)")
    s.add_argument("--out", required=True)
    s.add_argument("--hours", type=float, default=6.0)
    s.add_argument("--lag", type=float, default=3.0)
    s.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("research", help="walk-forward research run + evaluation report")
    r.add_argument("--data", default=None)
    r.add_argument("--start")
    r.add_argument("--end")
    r.add_argument("--hours", type=float)
    r.add_argument("--models")
    r.add_argument("--run-dir")
    r = sub.add_parser("report", help="re-render report.md from a run directory")
    r.add_argument("--run-dir", required=True)
    lab = sub.add_parser("lab", help="Alpha Laboratory (continuous research + promotion)")
    lab.add_argument("--data", default=None)
    lab.add_argument("--once", action="store_true")
    lab.add_argument("--hours", type=float, help="lookback window")
    lab.add_argument("--purpose", choices=["live", "paper"], default="live")
    t = sub.add_parser("trade", help="run the trader (paper by default)")
    t.add_argument("--mode", choices=["paper", "live"])
    t.add_argument("--i-understand-live-risk", action="store_true")
    i = sub.add_parser("integration-test", help="exchange integration test suite")
    i.add_argument("--place-test-order", action="store_true")
    i.add_argument("--latency-samples", type=int, default=0)
    sub.add_parser("status", help="health of collector, trader, lab and champion")
    u = sub.add_parser("ui", help="local web control panel (run as root via apexmind-ui.service)")
    u.add_argument("--bind", default="127.0.0.1")
    u.add_argument("--port", type=int, default=18787)
    v = sub.add_parser("validate-methodology", help="positive/negative synthetic controls")
    v.add_argument("--out", required=True)
    v.add_argument("--hours", type=float, default=6.0)
    v.add_argument("--lag", type=float, default=3.0)
    v.add_argument("--seed", type=int, default=5)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    if getattr(a, "data", "unset") is None:
        a.data = cfg.collector.data_dir
    handlers = {"collect": cmd_collect, "synthetic": cmd_synthetic, "research": cmd_research, "report": cmd_report,
                "lab": cmd_lab, "trade": cmd_trade, "integration-test": cmd_integration, "status": cmd_status,
                "ui": cmd_ui, "validate-methodology": cmd_validate}
    return handlers[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
