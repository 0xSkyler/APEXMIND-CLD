"""Champion/challenger model registry.

Layout (``LabConfig.registry_dir``)::

    bundles/<id>.pkl     pickled StrategyBundle (written by this process only)
    bundles/<id>.json    manifest: provenance, validation summary, sha256
    champion.json        pointer to the current champion bundle
    history.jsonl        append-only promotions, retirements, suspensions

Bundles are verified against their manifest hash before unpickling.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

from apexmind.strategy.decision import StrategyBundle
from apexmind.util import atomic_write_json


class RegistryError(RuntimeError):
    pass


class Registry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "bundles").mkdir(parents=True, exist_ok=True)

    def _log(self, kind: str, **detail) -> None:
        with open(self.root / "history.jsonl", "a") as fh:
            fh.write(json.dumps({"t": time.time(), "kind": kind, **detail}, default=str) + "\n")

    def save(self, bundle: StrategyBundle) -> str:
        blob = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(blob).hexdigest()
        bid = f"{bundle.name}-{time.strftime('%Y%m%dT%H%M%S')}-{digest[:8]}"
        (self.root / "bundles" / f"{bid}.pkl").write_bytes(blob)
        atomic_write_json(self.root / "bundles" / f"{bid}.json", {**bundle.manifest, "id": bid, "sha256": digest,
                                                                   "name": bundle.name})
        self._log("saved", id=bid)
        return bid

    def manifest(self, bid: str) -> dict:
        return json.loads((self.root / "bundles" / f"{bid}.json").read_text())

    def load(self, bid: str) -> StrategyBundle:
        man = self.manifest(bid)
        blob = (self.root / "bundles" / f"{bid}.pkl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != man["sha256"]:
            raise RegistryError(f"bundle {bid} failed integrity check")
        return pickle.loads(blob)  # noqa: S301 - local file written and hashed by this system

    def promote(self, bid: str, gate: dict) -> None:
        atomic_write_json(self.root / "champion.json", {"id": bid, "promoted_at": time.time(), "gate": gate})
        self._log("promoted", id=bid, gate=gate)

    def champion_id(self) -> str | None:
        p = self.root / "champion.json"
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("id")

    def load_champion(self) -> tuple[str, StrategyBundle] | None:
        bid = self.champion_id()
        return None if bid is None else (bid, self.load(bid))

    def retire(self, reason: str) -> None:
        bid = self.champion_id()
        p = self.root / "champion.json"
        if p.exists():
            p.unlink()
        self._log("retired", id=bid, reason=reason)

    def history(self) -> list[dict]:
        p = self.root / "history.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
