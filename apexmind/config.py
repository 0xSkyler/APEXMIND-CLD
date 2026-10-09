"""Typed configuration.

Every tunable lives here with a documented default. Values are loaded from a
YAML file and may be overridden with ``APEXMIND__SECTION__KEY=value``
environment variables. Secrets are never read from the YAML file; see
:func:`load_lighter_api_key`.
"""

from __future__ import annotations

import dataclasses
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class LighterConfig:
    profile: str = "mainnet"
    api_url: str = "https://mainnet.zklighter.elliot.ai"
    ws_url: str = "wss://mainnet.zklighter.elliot.ai/stream"
    chain_id: int = 304
    account_index: int = -1  # -1 = no account (market data only)
    api_key_index: int = -1
    # Public REST budget. Lighter weights endpoints; keep well under the
    # documented per-IP limit so the collector never starves the trader.
    rest_requests_per_minute: int = 50
    # Transactions per second we allow ourselves to send (all API keys).
    max_tx_per_second: float = 4.0
    ws_max_subscriptions_per_connection: int = 50
    ws_ping_interval_s: float = 20.0
    # "market": fees from market metadata (conservative default);
    # "config": the fractions below (set them to your account tier's fees,
    # verified by the integration test's realized-fee check).
    fee_source: str = "market"
    taker_fee: float = 0.0
    maker_fee: float = 0.0
    funding_interval_s: int = 3600


@dataclass
class ReferenceConfig:
    venue: str = "binance_usdm"
    ws_url: str = "wss://fstream.binance.com/stream"
    rest_url: str = "https://fapi.binance.com"
    depth_stream_ms: int = 100  # 100 or 250 or 500 for Binance diff depth
    snapshot_depth: int = 1000
    rest_weight_per_minute: int = 1200


@dataclass
class InstrumentConfig:
    # Start narrow (liquid, unambiguous mappings), expand after validation.
    initial_bases: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    expand_to_all_eligible: bool = False
    # Explicit overrides: lighter symbol -> reference symbol
    overrides: dict[str, str] = field(default_factory=dict)
    # Mapping is rejected when the price ratio after the contract multiplier
    # deviates more than this from 1.
    max_price_ratio_deviation: float = 0.02
    min_lighter_daily_quote_volume: float = 1_000_000.0


@dataclass
class CollectorConfig:
    data_dir: str = "data/raw"
    rotate_minutes: int = 60
    flush_interval_s: float = 1.0
    clock_probe_interval_s: float = 30.0
    snapshot_resync_minutes: int = 30
    health_path: str = "state/collector_health.json"


@dataclass
class FeatureConfig:
    grid_ms: int = 250
    return_horizons_s: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 15.0, 60.0])
    flow_horizons_s: list[float] = field(default_factory=lambda: [1.0, 5.0, 30.0])
    depth_levels: int = 10
    depth_band_bps: float = 10.0
    impact_notional_usd: float = 1_000.0
    basis_ewma_halflife_s: float = 600.0
    vol_ewma_halflife_s: float = 60.0
    regime_lookback_s: float = 3600.0
    max_staleness_ms: int = 3_000


@dataclass
class LabelConfig:
    horizons_s: list[float] = field(default_factory=lambda: [5.0, 15.0, 60.0])
    notional_usd: float = 1_000.0
    significant_slippage_bps: float = 5.0
    # "measured" uses recorded execution latencies; "prior" uses the
    # conservative prior below and is flagged in every report.
    latency_source: str = "measured"
    prior_latency_median_ms: float = 400.0
    prior_latency_sigma: float = 0.7  # lognormal shape
    # Passive (post-only at the touch) entry simulation: filled only when the
    # tape trades through our price or exhausts the displayed queue ahead.
    passive: bool = True
    passive_wait_s: float = 5.0
    seed: int = 7


@dataclass
class ResearchConfig:
    runs_dir: str = "runs"
    n_folds: int = 4
    min_train_hours: float = 24.0
    test_hours: float = 12.0
    calibration_fraction: float = 0.3
    embargo_s: float = 120.0
    bootstrap_reps: int = 2000
    block_minutes: float = 30.0
    confidence: float = 0.95
    models: list[str] = field(
        default_factory=lambda: [
            "ref_momentum",
            "lag_only",
            "flow_only",
            "lag_flow",
            "ridge",
            "gbm",
        ]
    )
    feature_selection: bool = True
    max_train_rows: int = 400_000
    min_calibration_trades: int = 50
    # Equity scenarios simulated for every model; the first is the primary
    # one used for equity metrics, the rest show small-account feasibility.
    initial_equities: list[float] = field(default_factory=lambda: [1000.0, 10.0])
    seed: int = 11


@dataclass
class StrategyConfig:
    # Trade only when the calibrated lower confidence bound of expected net
    # return exceeds this margin (fraction, after all costs).
    min_lcb_return: float = 0.0
    lcb_z: float = 1.645
    calibration_bins: int = 10
    min_calibration_trades_per_bin: int = 30
    max_data_skew_ms: int = 1_500
    one_position_per_market: bool = True


@dataclass
class ExecutionConfig:
    allow_passive: bool = True
    passive_max_wait_fraction: float = 0.5  # of signal horizon
    max_slippage_bps: float = 15.0
    order_expiry_s: int = 300


@dataclass
class PortfolioConfig:
    kelly_fraction: float = 0.25
    max_position_equity_fraction: float = 2.0  # notional / equity per position
    max_gross_leverage: float = 4.0
    max_margin_utilization: float = 0.5
    max_trade_loss_equity_fraction: float = 0.02  # ES-based per-trade loss budget
    correlation_halflife_s: float = 3600.0
    reserve_usd: float = 0.0


@dataclass
class RiskConfig:
    max_feed_staleness_ms: int = 3_000
    max_clock_uncertainty_ms: float = 250.0
    margin_warning_ratio: float = 0.5  # maintenance requirement / equity
    margin_reduce_ratio: float = 0.7
    max_drawdown_halt: float = 0.25  # fraction of high-water equity
    drawdown_scale_start: float = 0.10
    edge_min_trades: int = 30
    edge_window_trades: int = 200
    edge_suspend_after_reductions: int = 3


@dataclass
class LabConfig:
    interval_minutes: float = 360.0
    nice: int = 15
    max_memory_gb: float = 6.0
    max_threads: int = 2
    registry_dir: str = "runs/registry"
    promotion_min_oos_periods: int = 3
    promotion_min_trades: int = 200
    promotion_max_pbo: float = 0.3
    promotion_min_dsr: float = 0.95
    promotion_alpha: float = 0.05


@dataclass
class LiveConfig:
    mode: str = "paper"  # "paper" or "live"
    state_db: str = "state/apexmind.sqlite"
    heartbeat_path: str = "state/trader_heartbeat.json"
    integration_report: str = "state/integration_test.json"
    integration_max_age_hours: float = 24.0
    reconcile_interval_s: float = 10.0
    secrets_file: str = "/etc/apexmind/lighter_api_key"
    paper_equity: float = 1000.0
    flatten_on_shutdown: bool = True
    kill_file: str = "state/KILL"  # create this file to flatten and halt entries
    deadman_seconds: int = 60  # exchange cancels resting orders if not re-armed
    record_dir: str = "data/raw"  # where execution-latency records are written


@dataclass
class Config:
    lighter: LighterConfig = field(default_factory=LighterConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    instruments: InstrumentConfig = field(default_factory=InstrumentConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    lab: LabConfig = field(default_factory=LabConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(tp: Any, value: Any) -> Any:
    origin = get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value or {})
    if origin is list:
        (inner,) = get_args(tp) or (Any,)
        if isinstance(value, str):
            value = [v for v in value.split(",") if v]
        return [_coerce(inner, v) for v in value]
    if origin is dict:
        return dict(value)
    if tp is bool and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if tp in (int, float, str) and value is not None:
        return tp(value)
    return value


def _build(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v) for k, v in data.items()}
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str] | None = None, env: dict[str, str] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    env = dict(os.environ) if env is None else env
    for key, value in env.items():
        if not key.startswith("APEXMIND__"):
            continue
        parts = key[len("APEXMIND__") :].lower().split("__")
        if len(parts) != 2:
            continue
        data.setdefault(parts[0], {})[parts[1]] = value
    return _build(Config, data)


class SecretError(RuntimeError):
    pass


def load_lighter_api_key(cfg: LiveConfig, env: dict[str, str] | None = None) -> str:
    """Return the Lighter *API key* private key.

    The Ethereum wallet key is never needed for trading and must never be
    placed on the server: without it, L1-signed operations such as transfers
    cannot be produced from this machine. The API key is read from the
    ``APEXMIND_LIGHTER_API_KEY`` environment variable (set by systemd's
    ``LoadCredential``/``EnvironmentFile``) or from a file that must not be
    readable by group or others.
    """
    env = dict(os.environ) if env is None else env
    if env.get("APEXMIND_LIGHTER_API_KEY"):
        return env["APEXMIND_LIGHTER_API_KEY"].strip()
    cred_dir = env.get("CREDENTIALS_DIRECTORY")
    candidates = [Path(cred_dir) / "lighter_api_key"] if cred_dir else []
    candidates.append(Path(cfg.secrets_file))
    for p in candidates:
        if p.exists():
            mode = p.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise SecretError(f"{p} must not be accessible by group/others (chmod 600)")
            return p.read_text().strip()
    raise SecretError("Lighter API key not configured")
