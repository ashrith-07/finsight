"""Report housekeeping helpers."""

from pathlib import Path
import os
import time

import pytest

from src.reports_service import (
    cleanup_reports,
    delete_report,
    is_safe_report_filename,
    list_reports,
)


def test_is_safe_report_filename() -> None:
    assert is_safe_report_filename("portfolio_x_20260101_120000.md") is True
    assert is_safe_report_filename("x.pdf") is True
    assert is_safe_report_filename("../evil.md") is False
    assert is_safe_report_filename("a/b.md") is False
    assert is_safe_report_filename(".gitkeep") is False


def test_list_and_delete_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "ok_20260101_000000.md"
    p.write_text("# hi\n", encoding="utf-8")
    rows = list_reports(tmp_path)
    assert len(rows) == 1
    assert rows[0]["name"] == "ok_20260101_000000.md"
    assert delete_report(tmp_path, "ok_20260101_000000.md") is True
    assert not p.exists()


def test_cleanup_age(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPORT_RETENTION_HOURS", "1")
    old = tmp_path / "old_20200101_000000.md"
    old.write_text("x", encoding="utf-8")
    old_t = time.time() - 7200.0
    os.utime(old, (old_t, old_t))
    stats = cleanup_reports(tmp_path)
    assert stats["deleted"] >= 1
    assert not old.exists()
