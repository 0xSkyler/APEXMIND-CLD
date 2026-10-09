"""Static guarantee: the codebase never calls withdrawal/transfer paths."""

import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "apexmind"
FORBIDDEN = re.compile(r"withdraw|transfer|fastwithdraw|change_api_key|approve_integrator|create_sub_account|"
                       r"mint_shares|burn_shares|stake_assets|unstake_assets|public_pool|eth_private_key", re.I)


def _called_names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                yield f.attr, node.lineno
            elif isinstance(f, ast.Name):
                yield f.id, node.lineno
        elif isinstance(node, ast.Attribute):
            yield node.attr, node.lineno


def test_no_fund_moving_calls_anywhere():
    offenders = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for name, line in _called_names(tree):
            if FORBIDDEN.search(name):
                offenders.append(f"{path.relative_to(PKG.parent)}:{line} {name}")
    assert not offenders, "fund-moving API referenced:\n" + "\n".join(offenders)


def test_gateway_surface_is_trading_only():
    from apexmind.venues.lighter.trading import LighterGateway, PaperGateway

    allowed = {"start", "close", "auth_token", "create_order", "cancel_order", "cancel_all", "schedule_cancel_all",
               "update_leverage"}
    for cls in (LighterGateway, PaperGateway):
        public = {n for n in vars(cls) if not n.startswith("_") and callable(getattr(cls, n))}
        assert public - allowed - {"poll", "on_trade"} == set(), public - allowed


def test_eth_wallet_key_never_loaded():
    src = (PKG / "config.py").read_text()
    assert "load_lighter_api_key" in src
    for path in PKG.rglob("*.py"):
        assert "eth_private_key" not in path.read_text()
