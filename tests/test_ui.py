import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from apexmind.ui.server import Panel, PanelError, make_handler

CONFIG = """# comment kept
lighter:
  profile: mainnet
  account_index: -1        # set for account/trading features
  api_key_index: -1
reference:
  venue: binance_usdm
"""
API_KEY = "ab" * 40


class FakeSystemd:
    def __init__(self):
        self.calls, self.active = [], set()

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[:2] == ["systemctl", "is-active"]:
            return (0, "active") if argv[2] in self.active else (3, "inactive")
        if argv[:2] == ["systemctl", "is-enabled"]:
            return 0, "enabled"
        return 0, ""


@pytest.fixture
def panel(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG)
    (tmp_path / "work").mkdir()
    return Panel(str(cfg), key_path=str(tmp_path / "key"), token_path=str(tmp_path / "token"),
                 dropin_dir=str(tmp_path / "dropin"), workdir=str(tmp_path / "work"), runner=FakeSystemd())


def test_credentials_saved_privately_and_config_comments_kept(panel):
    panel.set_credentials("180399", "4", API_KEY)
    assert panel.key_path.read_text().strip() == API_KEY
    assert oct(panel.key_path.stat().st_mode & 0o777) == "0o600"
    text = panel.config_path.read_text()
    assert "account_index: 180399        # set for account/trading features" in text
    assert "api_key_index: 4" in text and text.startswith("# comment kept")
    # index-only update keeps the saved key
    panel.set_credentials(180399, 5, "")
    assert panel.key_path.read_text().strip() == API_KEY and "api_key_index: 5" in panel.config_path.read_text()
    st = panel.status()
    assert st["credentials"] == {"account_index": 180399, "api_key_index": 5, "key": st["credentials"]["key"]}
    assert st["credentials"]["key"]["set"] and API_KEY not in json.dumps(st)


def test_rejects_wallet_key_bad_key_and_reserved_index(panel):
    with pytest.raises(PanelError, match="WALLET"):
        panel.set_credentials(1, 4, "0x" + "cd" * 32)
    with pytest.raises(PanelError):
        panel.set_credentials(1, 4, "not-a-key")
    with pytest.raises(PanelError, match="reserved"):
        panel.set_credentials(1, 2, API_KEY)
    assert not panel.key_path.exists()


def test_services_mode_and_kill(panel):
    panel.service("collector", "start")
    assert ["systemctl", "enable", "--now", "apexmind-collector"] in panel.run.calls
    with pytest.raises(PanelError):
        panel.service("sshd", "stop")
    with pytest.raises(PanelError, match="LIVE"):
        panel.set_mode("live", "yes")
    assert panel.mode() == "paper"
    panel.run.active.add("apexmind-trader")
    panel.set_mode("live", "LIVE")
    assert panel.mode() == "live" and "--i-understand-live-risk" in panel.dropin.read_text()
    assert ["systemctl", "restart", "apexmind-trader"] in panel.run.calls
    panel.set_mode("paper")
    assert panel.mode() == "paper"
    panel.set_kill(True)
    assert panel.kill_path().exists() and panel.status()["kill"]
    panel.set_kill(False)
    assert not panel.kill_path().exists()


def test_http_requires_local_host_token_and_custom_header(panel):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(panel, set()))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(panel, {f"127.0.0.1:{port}"})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def req(path, data=None, headers=None, host=None):
        r = urllib.request.Request(base + path, data=data, headers=headers or {})
        if host:
            r.add_header("Host", host)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    try:
        assert req("/", host="evil.example:1234")[0] == 403  # DNS rebinding
        assert "Access token" in req("/")[1]
        assert req("/api/status")[0] == 401
        cookie = {"Cookie": f"apexmind_ui={panel.token()}"}
        assert req("/api/status", headers=cookie)[0] == 200
        body = json.dumps({"account_index": 7, "api_key_index": 4, "private_key": API_KEY}).encode()
        jh = {**cookie, "Content-Type": "application/json"}
        assert req("/api/credentials", body, jh)[0] == 401  # missing X-Apexmind header
        code, out = req("/api/credentials", body, {**jh, "X-Apexmind": "1"})
        assert code == 200 and "saved" in out and API_KEY not in out
        assert req("/api/status", headers=cookie)[1].count(API_KEY) == 0
    finally:
        httpd.shutdown()


def test_trader_start_refused_without_champion_and_status_is_read_only(panel):
    with pytest.raises(PanelError, match="No strategy"):
        panel.service("trader", "start")
    panel.service("trader", "stop")  # stopping is always allowed
    panel.status()
    assert not (panel.workdir / "runs").exists()  # no root-owned dirs created by polling
    reg = panel.workdir / "runs" / "registry"
    reg.mkdir(parents=True)
    (reg / "champion.json").write_text('{"id": "ridge-1"}')
    assert panel.status()["health"]["champion"] == "ridge-1"
    panel.service("trader", "start")


def test_lab_reports_waiting_for_data(tmp_path):
    from apexmind.config import Config
    from apexmind.data.synthetic import SyntheticSpec, generate
    from apexmind.lab.laboratory import AlphaLab

    raw = tmp_path / "raw"
    generate(SyntheticSpec(hours=0.2, seed=3), str(raw))
    cfg = Config()
    cfg.research.runs_dir = str(tmp_path / "runs")
    cfg.lab.registry_dir = str(tmp_path / "runs" / "registry")
    out = AlphaLab(cfg, str(raw)).run_cycle()
    assert out["status"] == "waiting_for_data" and 0.1 < out["recorded_hours"] < 0.3
    assert json.loads((tmp_path / "runs" / "lab_status.json").read_text())["first_run_at_hours"] == 37.0
