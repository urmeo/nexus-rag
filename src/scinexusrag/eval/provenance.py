"""Identify the source and runtime used for a benchmark."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PACKAGES = ("torch", "sentence-transformers", "transformers", "numpy", "datasets")


def evaluation_provenance(source_dir: Path | None = None) -> dict[str, Any]:
    source_dir = source_dir or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*.py")):
        digest.update(path.relative_to(source_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    commit = None
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "source_sha256": digest.hexdigest(),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
