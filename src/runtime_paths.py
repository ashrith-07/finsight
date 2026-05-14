"""Resolve writable paths for reports, Agno memory, and other local state.

``DATA_DIR`` is the canonical base (e.g. ``/app/data`` in Docker, or project root
locally). ``REPORTS_DIR`` and ``FINSIGHT_MEMORY_DB`` may be absolute or relative
to ``DATA_DIR`` when set.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    raw = os.environ.get("DATA_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    return project_root()


def reports_dir() -> Path:
    raw = os.environ.get("REPORTS_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = data_dir() / p
    else:
        p = data_dir() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def default_agno_memory_db_path() -> str:
    explicit = os.environ.get("FINSIGHT_MEMORY_DB", "").strip()
    if explicit:
        ep = Path(explicit).expanduser()
        if not ep.is_absolute():
            ep = data_dir() / ep
        ep.parent.mkdir(parents=True, exist_ok=True)
        return str(ep)
    p = data_dir() / ".agno_memory.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)
