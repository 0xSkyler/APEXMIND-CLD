"""Live gateway against the official Lighter signer, fully offline.

The signer's network call (``send_tx``) is replaced so nothing leaves the
machine; signing, nonce management, integer encoding and response handling
are the real SDK code paths.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

lighter = pytest.importorskip("lighter")

from apexmind.config import LighterConfig  # noqa: E402
from apexmind.core.clock import Clock  # noqa: E402
from apexmind.venues.lighter.markets import LighterMarket  # noqa: E402
from apexmind.venues.lighter.trading import LighterGateway  # noqa: E402


def market():
    return LighterMarket(0, "ETH", "active", 0.0002, 0.00002, 0.005, 10.0, 4, 2, 0.05, 0.02, 0.012)


async def _run():
    from lighter.signer_client import create_api_key

    priv, _pub, err = create_api_key()
    assert err is None
    cfg = LighterConfig(api_url="https://testnet.zklighter.elliot.ai", chain_id=300, account_index=123,
                        api_key_index=3, max_tx_per_second=100)
    gw = LighterGateway(cfg, priv, Clock())
    await gw.start(verify=False)  # key registration check needs the network
    sent = []

    async def fake_send(tx_type, tx_info):
        sent.append((tx_type, json.loads(tx_info)))
        return SimpleNamespace(code=200, tx_hash=f"0x{len(sent):04x}", predicted_execution_time_ms=12,
                               volume_quota_remaining=99, message=None)

    async def fake_nonce(api_key=None):
        return 3, 100 + len(sent)

    gw._signer.send_tx = fake_send
    gw._signer.nonce_manager.async_next_nonce = fake_nonce
    m = market()
    r1 = await gw.create_order(m, 42, 1, 0.0123, 3000.555, "post_only")
    r2 = await gw.create_order(m, 43, -1, 0.02, 2999.444, "ioc", reduce_only=True)
    r3 = await gw.cancel_order(m, 777)
    r4 = await gw.schedule_cancel_all(1_900_000_000_000)
    await gw.close()
    return sent, (r1, r2, r3, r4), gw


def test_signed_transactions_encode_orders_correctly():
    sent, results, gw = asyncio.run(_run())
    assert all(r.ok for r in results)
    (t1, o1), (t2, o2), (t3, c), (t4, ca) = sent
    assert (t1, t2, t3, t4) == (14, 14, 15, 16)
    # buy post-only: price rounded down, size floored to the lot
    assert o1["Price"] == 300055 and o1["BaseAmount"] == 123 and o1["IsAsk"] == 0
    assert o1["TimeInForce"] == 2 and o1["ReduceOnly"] == 0 and o1["OrderExpiry"] > 0
    # sell IOC reduce-only: price rounded up, no expiry
    assert o2["Price"] == 299945 and o2["IsAsk"] == 1 and o2["TimeInForce"] == 0 and o2["ReduceOnly"] == 1
    assert o2["OrderExpiry"] == 0
    assert c["Index"] == 777 or 777 in c.values()
    assert all(tx["AccountIndex"] == 123 and tx["ApiKeyIndex"] == 3 for _, tx in sent)
    assert gw.quota_remaining == 99
    assert gw._key == ""  # the gateway does not keep a second copy of the key
