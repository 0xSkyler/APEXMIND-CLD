"""Small shared utilities: atomic writes, hashing, provenance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, default=_json_default, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def config_hash(cfg) -> str:
    return stable_hash(cfg.to_dict())


def git_revision(cwd: str | os.PathLike | None = None) -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=cwd or Path(__file__).resolve().parent.parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=5, cwd=cwd or Path(__file__).resolve().parent.parent).stdout.strip()
        return (rev or "unknown") + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def file_digest(paths: list[Path]) -> str:
    """Content hash of a set of data files (order-independent)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]
