import copy

import numpy as np
import pytest

from apexmind.config import Config
from apexmind.data.synthetic import SymbolSpec, SyntheticSpec, generate
from apexmind.lab.laboratory import AlphaLab, psi
from apexmind.lab.promotion import promotion_gate
from apexmind.lab.registry import Registry, RegistryError
from apexmind.research.policy import Policy
from apexmind.strategy.decision import StrategyBundle


def passing_results(kind="recorded", latency="measured"):
    return {
        "meta": {"dataset": {"data_kind": kind, "latency": {"source": latency}}, "results_hash": "abc",
                 "primary_equity": "1000"},
        "folds": [{}, {}, {}],
        "champion": {"selected": {"model": "ridge", "mode": "aggressive"}, "beats_all_baselines_after_costs": True,
                     "verdict": "credible"},
        "models": {"ridge": {"modes": {"aggressive": {"equity_scenarios": {
            "10": {"trades": {"n_trades": 5}},
            "1000": {"trades": {"n_trades": 500, "mean_net_bps_ci": [0.5, 2.0]}}}}}}},
        "comparisons": {"aggressive": {"spa_vs_no_trading": {"p_value": 0.01}, "pbo": {"pbo": 0.1},
                                       "deflated_sharpe": {"ridge": {"dsr": 0.99}}}},
    }


def test_gate_passes_only_with_complete_evidence():
    cfg = Config()
    g = promotion_gate(passing_results(), cfg, "live", reproduced_hash="abc")
    assert g.passed, g.failures()
    # the primary ($1000) scenario is used, not whichever key sorts first
    assert "500 OOS trades" in g.checks["min_trades"]["detail"]
    for mutate, check in [
        (lambda r: r["meta"]["dataset"].update(data_kind="synthetic"), "data_recorded"),
        (lambda r: r["meta"]["dataset"].update(latency={"source": "prior"}), "latency_measured"),
        (lambda r: r["comparisons"]["aggressive"]["pbo"].update(pbo=0.6), "pbo"),
        (lambda r: r["comparisons"]["aggressive"]["deflated_sharpe"]["ridge"].update(dsr=0.5), "deflated_sharpe"),
        (lambda r: r["champion"].update(beats_all_baselines_after_costs=False), "beats_baselines"),
        (lambda r: r["models"]["ridge"]["modes"]["aggressive"]["equity_scenarios"]["1000"]["trades"].update(
            mean_net_bps_ci=[-0.1, 2.0]), "net_ci_positive"),
    ]:
        r = passing_results()
        mutate(r)
        g = promotion_gate(r, cfg, "live", reproduced_hash="abc")
        assert not g.passed and not g.checks[check]["ok"], check
    assert not promotion_gate(passing_results(), cfg, "live", reproduced_hash="zzz").passed
    assert not promotion_gate(passing_results(), cfg, "live", "abc", None, champion_exists=True).passed
    beats = {"mean": 0.001, "p_value": 0.01}
    assert promotion_gate(passing_results(), cfg, "live", "abc", beats, champion_exists=True).passed
    # synthetic evidence can qualify for paper trading only
    assert promotion_gate(passing_results("synthetic", "prior"), cfg, "paper", "abc").passed
    r = passing_results()
    r["champion"] = {"selected": None, "verdict": "NO MODEL"}
    assert not promotion_gate(r, cfg, "paper", "abc").passed


def test_registry_integrity_and_lifecycle(tmp_path):
    reg = Registry(tmp_path)
    b = StrategyBundle("m", object(), Policy("m", {"long": None, "short": None}, {}), ["f"], {}, {}, 1.0,
                       {"purpose": "paper"})
    bid = reg.save(b)
    reg.promote(bid, {"passed": True})
    cid, loaded = reg.load_champion()
    assert cid == bid and loaded.name == "m"
    p = tmp_path / "bundles" / f"{bid}.pkl"
    p.write_bytes(p.read_bytes() + b"tamper")
    with pytest.raises(RegistryError):
        reg.load(bid)
    reg.retire("test")
    assert reg.champion_id() is None
    assert [h["kind"] for h in reg.history()] == ["saved", "promoted", "retired"]


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(1.0, 1.0, size=5000)) > 0.2


def test_lab_cycle_reproducible_and_paper_only(tmp_path):
    spec = SyntheticSpec(hours=2.0, seed=4, symbols=[SymbolSpec(), SymbolSpec(symbol="SOL", market_id=1, price=150.0,
                                                                                 tick=0.01)])
    generate(spec, str(tmp_path / "raw"))
    cfg = Config()
    cfg.labels.latency_source = "prior"
    cfg.research.models = ["lag_only", "ridge"]
    cfg.research.min_train_hours = 0.6
    cfg.research.test_hours = 0.3
    cfg.research.n_folds = 3
    cfg.research.block_minutes = 2.0
    cfg.research.bootstrap_reps = 200
    cfg.research.runs_dir = str(tmp_path / "runs")
    cfg.lab.registry_dir = str(tmp_path / "registry")
    cfg.lab.promotion_min_trades = 30
    cycle = AlphaLab(cfg, str(tmp_path / "raw"), purpose="paper").run_cycle()
    assert cycle["data_kind"] == "synthetic"
    cv = cycle["results"]["cvifa"]
    assert cv["gate"]["checks"]["reproducible"]["ok"], cv["gate"]["checks"]["reproducible"]
    assert "deferred" in cycle["results"]["liquidity_shock_reversion"]["status"] or cv.get("promoted")
    reg = Registry(cfg.lab.registry_dir)
    if cv.get("promoted"):
        man = reg.manifest(cv["promoted"])
        assert man["purpose"] == "paper" and man["data_kind"] == "synthetic"
    # the same evidence must never pass the live gate
    live_cycle_cfg = copy.deepcopy(cfg)
    lab_live = AlphaLab(live_cycle_cfg, str(tmp_path / "raw"), purpose="live")
    import json
    res = json.loads((tmp_path / "runs").joinpath(cv["run_dir"].split("/")[-1], "results.json").read_text())
    g = promotion_gate(res, live_cycle_cfg, "live", res["meta"]["results_hash"])
    assert not g.passed and not g.checks["data_recorded"]["ok"]
    assert lab_live.registry.root.exists()
