#!/usr/bin/env bash
# Install APEX MIND on Ubuntu (22.04+) as unprivileged systemd services.
# Usage: sudo ./deploy/install.sh   (from a checkout of this repository)
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)"; exit 1; }
SRC="$(cd "$(dirname "$0")/.." && pwd)"

apt-get update -y
apt-get install -y python3 python3-venv chrony sqlite3
systemctl enable --now chrony   # clock discipline; status is recorded by the collector

id apexmind &>/dev/null || useradd --system --home /var/lib/apexmind --shell /usr/sbin/nologin apexmind
install -d -o apexmind -g apexmind -m 0750 /var/lib/apexmind /var/lib/apexmind/data /var/lib/apexmind/state /var/lib/apexmind/runs
install -d -o root -g apexmind -m 0750 /etc/apexmind
[[ -f /etc/apexmind/config.yaml ]] || install -o root -g apexmind -m 0640 "$SRC/configs/default.yaml" /etc/apexmind/config.yaml

python3 -m venv /opt/apexmind/venv
/opt/apexmind/venv/bin/pip install --upgrade pip
/opt/apexmind/venv/bin/pip install "$SRC[live]"

install -m 0644 "$SRC"/deploy/systemd/apexmind-*.service /etc/systemd/system/
systemctl daemon-reload

# Operator command: runs as apexmind from /var/lib/apexmind, where the services
# resolve the config's relative paths (state/, runs/, data/), so it works from
# any directory. It has no credential; integration-test uses systemd-run below.
cat > /usr/local/bin/apexmind <<'WRAP'
#!/bin/sh
cd /var/lib/apexmind && exec sudo -u apexmind /opt/apexmind/venv/bin/apexmind --config /etc/apexmind/config.yaml "$@"
WRAP
chmod 0755 /usr/local/bin/apexmind
systemctl enable --now apexmind-collector.service apexmind-lab.service

cat <<'MSG'
Installed. The collector and the Alpha Lab are running.

Next steps (in order):
 1. Let the collector record data; check:   sudo apexmind status
 2. For account features, store ONLY the Lighter API key (never the wallet key):
      sudo install -o root -g root -m 0600 /dev/stdin /etc/apexmind/lighter_api_key   (paste key, Ctrl-D)
    and set lighter.account_index / api_key_index in /etc/apexmind/config.yaml.
 3. Run the exchange integration test (as apexmind, with the credential):
      sudo systemd-run --pty --uid=apexmind -p LoadCredential=lighter_api_key:/etc/apexmind/lighter_api_key \
        -p WorkingDirectory=/var/lib/apexmind /opt/apexmind/venv/bin/apexmind --config /etc/apexmind/config.yaml \
        integration-test --place-test-order --latency-samples 30
 4. Paper trading starts once the lab has promoted a champion:  systemctl enable --now apexmind-trader
 5. Live trading requires a champion validated on recorded data with measured latency and a recent
    passing integration test; the trader refuses otherwise.
MSG
