"""Report artefact listing, safe deletion, and retention housekeeping."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.logging_config import get_logger

logger = get_logger("reports_service")

_ALLOWED_SUFFIXES = frozenset({".md", ".pdf"})
_RESERVED = frozenset({".gitkeep", "readme.md", "readme.txt"})


def is_safe_report_filename(name: str) -> bool:
    if not name or "/" in name or "\\" in name or ".." in name:
        return False
    low = name.lower()
    if low in _RESERVED:
        return False
    return Path(name).suffix.lower() in _ALLOWED_SUFFIXES


def list_reports(reports_root: Path) -> list[dict[str, str | int]]:
    out: list[dict[str, str | int]] = []
    if not reports_root.is_dir():
        return out
    for entry in reports_root.iterdir():
        if not entry.is_file():
            continue
        if not is_safe_report_filename(entry.name):
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        out.append(
            {
                "name": entry.name,
                "bytes": int(st.st_size),
                "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    out.sort(key=lambda r: r["name"])
    return out


def _retention_hours() -> float:
    raw = os.environ.get("REPORT_RETENTION_HOURS", "72")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 72.0


def _max_total_bytes() -> int | None:
    raw = os.environ.get("REPORT_MAX_TOTAL_MB", "256").strip()
    if raw in {"", "0", "off", "none"}:
        return None
    try:
        return max(1, int(float(raw) * 1024 * 1024))
    except ValueError:
        return 256 * 1024 * 1024


def _max_files() -> int | None:
    raw = os.environ.get("REPORT_MAX_FILES", "500").strip()
    if raw in {"", "0", "off", "none"}:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return 500


def _gather(reports_root: Path) -> list[tuple[Path, float, int]]:
    rows: list[tuple[Path, float, int]] = []
    if not reports_root.is_dir():
        return rows
    for entry in reports_root.iterdir():
        if not entry.is_file() or not is_safe_report_filename(entry.name):
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        rows.append((entry, st.st_mtime, st.st_size))
    rows.sort(key=lambda r: r[1])
    return rows


def cleanup_reports(reports_root: Path) -> dict[str, int]:
    """Age, total size, and file-count caps — oldest artefacts removed first."""
    if not reports_root.is_dir():
        return {"deleted": 0, "freed_bytes": 0}

    retention_s = _retention_hours() * 3600.0
    now = time.time()
    max_bytes = _max_total_bytes()
    max_files = _max_files()
    deleted = 0
    freed = 0

    # 1) Age-based deletion
    for path, mtime, size in _gather(reports_root):
        if now - mtime <= retention_s:
            continue
        try:
            path.unlink()
            deleted += 1
            freed += size
        except OSError as e:
            logger.warning("report_retention_age_delete_failed path=%s err=%s", path, e)

    rows = _gather(reports_root)

    # 2) File count cap (drop oldest first)
    if max_files is not None and len(rows) > max_files:
        for path, _, size in rows[: len(rows) - max_files]:
            try:
                path.unlink()
                deleted += 1
                freed += size
            except OSError as e:
                logger.warning("report_retention_count_delete_failed path=%s err=%s", path, e)

    rows = _gather(reports_root)

    # 3) Total size cap (drop oldest until under budget)
    if max_bytes is not None:
        total = sum(s for _, _, s in rows)
        i = 0
        while total > max_bytes and i < len(rows):
            path, _, size = rows[i]
            i += 1
            try:
                if path.exists():
                    path.unlink()
                    deleted += 1
                    freed += size
                    total -= size
            except OSError as e:
                logger.warning("report_retention_size_delete_failed path=%s err=%s", path, e)

    if deleted:
        logger.info(
            "report_retention_run",
            extra={"deleted": deleted, "freed_bytes": freed},
        )
    return {"deleted": deleted, "freed_bytes": freed}


def retention_policy() -> dict[str, str | float | int | None]:
    max_mb_raw = os.environ.get("REPORT_MAX_TOTAL_MB", "256").strip()
    max_files_raw = os.environ.get("REPORT_MAX_FILES", "500").strip()
    return {
        "retention_hours": _retention_hours(),
        "max_total_mb": None if max_mb_raw.lower() in {"off", "none", ""} else max_mb_raw,
        "max_files": None if max_files_raw.lower() in {"off", "none", ""} else max_files_raw,
        "cleanup_interval_sec": os.environ.get("REPORT_CLEANUP_INTERVAL_SEC", "3600"),
    }


def delete_report(reports_root: Path, name: str) -> bool:
    if not is_safe_report_filename(name):
        return False
    path = reports_root / name
    try:
        if path.is_file() and path.resolve().parent == reports_root.resolve():
            path.unlink()
            return True
    except OSError:
        return False
    return False
