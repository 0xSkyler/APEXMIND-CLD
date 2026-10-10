# Operations runbook (Ubuntu)

## Install

```bash
git clone <this repo> && cd APEXMIND-CLD
sudo ./deploy/install.sh
```

This creates the unprivileged `apexmind` user, `/opt/apexmind/venv`,
`/var/lib/apexmind` (data, state, runs) and `/etc/apexmind/config.yaml`,
installs chrony, and enables `apexmind-collector` and `apexmind-lab`. All
services run under systemd with `Restart=always`, so no SSH session is needed.

The host must be able to reach `mainnet.zklighter.elliot.ai` and the
reference venue (`fstream.binance.com`, `fapi.binance.com`). Binance does not
serve some jurisdictions. Pick a hosting region where both are reachable.

## Control panel

`apexmind-ui` serves a local web panel at `http://127.0.0.1:18787`. Open it in a
browser on the server itself (for example over RustDesk or VNC). It covers:

* saving the Lighter account index, API key index and API private key;
* starting, stopping and restarting the collector, lab and trader;
* switching the trader between paper and live;
* the kill switch;
* running the integration test;
* service logs.

The access token is in `/etc/apexmind/ui_token` (`sudo cat` it). The panel
listens on 127.0.0.1 only, rejects other `Host` headers, never shows the saved
key, and refuses 64-hex-character keys, since those are wallet keys. To change
the port, run `systemctl edit apexmind-ui` and set
`Environment=APEXMIND_UI_PORT=NNNNN`. To reach it from another computer, use an
SSH tunnel (`ssh -L 18787:127.0.0.1:18787 user@vps`). Do not expose the port
publicly.

## Credentials

* Put **only the Lighter API key** on the host:
  `sudo install -m 0600 /dev/stdin /etc/apexmind/lighter_api_key`.
  systemd passes it to the trader with `LoadCredential`.
* Never place the Ethereum wallet private key on the host. Without it, L1
  operations such as transfers cannot be produced from this machine. The code
  base has no withdrawal or transfer calls, and a test enforces that.
* Set `lighter.account_index` and `lighter.api_key_index` in the config.
  Prefer a dedicated sub-account that holds only the capital you intend to risk.

## Lifecycle

1. **Collect** for long enough to cover several market regimes. The default
   protocol needs at least 24 h of training plus 3 × 12 h test periods; a week
   or more is far better. Check with `sudo apexmind status`
   (installed wrapper: runs as `apexmind` from `/var/lib/apexmind`, where the
   config's relative paths resolve).
2. **Integration test** on the host (as `apexmind`, with the credential):
   `apexmind integration-test --place-test-order --latency-samples 30`.
   The post-only order is placed 10% away from the touch and cannot fill.
   Latency calibration sends 30 minimum-size IOC round trips (cost ≈ spread +
   fees on the minimum notional each). The fills are timed on the public tape
   and stored as measured latency.
3. **Research**: the lab runs automatically (`apexmind-lab`). Each cycle writes
   `runs/<hypothesis>-<time>/report.md`, `results.json`, `trades.csv`, a
   reproduction run, and `runs/lab_status.json`.
4. **Paper trading**: `systemctl enable --now apexmind-trader` (paper by
   default) once a champion exists. Evaluate paper results against the report.
5. **Live trading** requires all of:
   * a champion promoted for `live` (recorded data, measured latency, full gate);
   * a passing integration test less than 24 h old for the same account that
     exercised order placement and cancellation;
   * `systemctl edit apexmind-trader` with
     `Environment=APEXMIND_TRADE_ARGS=--mode live --i-understand-live-risk`.
   The trader refuses to start otherwise.

## Controls

| Control | How |
|---|---|
| Halt and flatten | `touch /var/lib/apexmind/state/KILL` (remove the file to resume) |
| Stop service | `systemctl stop apexmind-trader` (flattens on SIGTERM by default) |
| Dead-man switch | The trader re-arms an exchange-side scheduled cancel-all (60 s) every reconcile cycle |
| Margin protection | Maintenance/equity ≥ 0.5 blocks entries; ≥ 0.7 flattens (reduce-only) |
| Drawdown | Size scales down from 10% drawdown, and trading halts and flattens at 25% |
| Edge decay | Live net returns below the calibrated LCB halve size, then suspend |
| Unknown positions | Exchange positions not in local state block that market; they are never auto-closed |

## Monitoring

* `sudo apexmind status` summarizes the collector, trader heartbeat, lab and champion.
* `journalctl -u apexmind-trader -f` shows the live log.
* SQLite state `state/apexmind.sqlite` holds orders, fills, positions, every
  signal decision (approved or rejected, with reason), latency records, equity
  and events (halts, mismatches, recoveries).

## Recovery

After a crash or reboot, systemd restarts the services. The trader reloads
state. In live mode it blocks entries until the first reconciliation has
resolved every in-flight order against the exchange. Open positions continue
to be exited, with exponential backoff between exit attempts.

## Resource use

The lab runs with `Nice=15`, `CPUQuota=200%` and `MemoryMax=6G`, with idle I/O
priority. Datasets are cached per UTC day, so each cycle rebuilds only the
newest day. The collector and trader run at higher priority than research.
