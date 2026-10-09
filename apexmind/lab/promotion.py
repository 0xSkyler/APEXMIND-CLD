"""Automated promotion gate for challenger strategies.

A challenger is promoted only if *every* check passes. For live use the
evidence must come from recorded market data with measured execution
latency; synthetic data or prior latency can at most qualify a bundle for
paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass

from apexmind.config import Config


@dataclass
class GateResult:
    passed: bool
    purpose: str
    checks: dict[str, dict]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "purpose": self.purpose, "checks": self.checks}

    def failures(self) -> list[str]:
        return [f"{k}: {v['detail']}" for k, v in self.checks.items() if not v["ok"]]


def promotion_gate(results: dict, cfg: Config, purpose: str = "live", reproduced_hash: str | None = None,
                   vs_champion: dict | None = None, champion_exists: bool = False) -> GateResult:
    lab = cfg.lab
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    ds = results["meta"]["dataset"]
    if purpose == "live":
        check("data_recorded", ds.get("data_kind") == "recorded", f"data kind {ds.get('data_kind')}")
        lat = ds.get("latency", {}).get("source", "")
        check("latency_measured", lat == "measured", f"latency source {lat}")
    champ = results.get("champion", {})
    sel = champ.get("selected")
    check("credible_selection", sel is not None, champ.get("verdict", ""))
    if sel is None:
        return GateResult(False, purpose, checks)
    model, mode = sel["model"], sel["mode"]
    primary = results["models"][model]["modes"][mode]["equity_scenarios"][results["meta"]["primary_equity"]]
    t = primary["trades"]
    n_folds = len(results["folds"])
    check("oos_periods", n_folds >= lab.promotion_min_oos_periods, f"{n_folds} OOS periods (min {lab.promotion_min_oos_periods})")
    check("min_trades", t.get("n_trades", 0) >= lab.promotion_min_trades,
          f"{t.get('n_trades', 0)} OOS trades (min {lab.promotion_min_trades})")
    lo = (t.get("mean_net_bps_ci") or [None])[0]
    check("net_ci_positive", lo is not None and lo > 0, f"net mean CI lower bound {lo} bps")
    cmp = results["comparisons"].get(mode, {})
    spa = cmp.get("spa_vs_no_trading", {}).get("p_value")
    check("spa_vs_no_trading", spa is not None and spa < lab.promotion_alpha, f"SPA p={spa}")
    check("beats_baselines", bool(champ.get("beats_all_baselines_after_costs")),
          "paired tests vs every baseline (5%)")
    pbo = cmp.get("pbo", {}).get("pbo")
    check("pbo", pbo is not None and pbo <= lab.promotion_max_pbo, f"PBO={pbo} (max {lab.promotion_max_pbo})")
    dsr = cmp.get("deflated_sharpe", {}).get(model, {}).get("dsr")
    check("deflated_sharpe", dsr is not None and dsr >= lab.promotion_min_dsr, f"DSR={dsr} (min {lab.promotion_min_dsr})")
    if reproduced_hash is not None:
        h = results["meta"]["results_hash"]
        check("reproducible", reproduced_hash == h, f"rerun hash {reproduced_hash} vs {h}")
    else:
        check("reproducible", False, "no reproduction run")
    if champion_exists:
        if vs_champion is None:
            check("beats_champion", False, "too few periods unseen by the champion to compare")
        else:
            ok = vs_champion.get("p_value", 1.0) < lab.promotion_alpha and vs_champion.get("mean", 0) > 0
            check("beats_champion", ok,
                  f"paired vs champion: mean={vs_champion.get('mean')} p={vs_champion.get('p_value')}")
    passed = all(v["ok"] for v in checks.values())
    return GateResult(passed, purpose, checks)
