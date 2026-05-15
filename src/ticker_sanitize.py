"""Reject junk symbols Agno sometimes invents (e.g. ``ALEX_PORTFOLIO``) before Yahoo calls."""

from __future__ import annotations

import re

_TICKER_OK = re.compile(r"^[A-Z0-9.\-]{1,15}$")

_JUNK_SUBSTR = (
    "PORTFOLIO",
    "HOLDING",
    "HOLDINGS",
    "POSITION",
    "POSITIONS",
    "USER",
    "CLIENT",
    "ACCOUNT",
    "BENCHMARK",
    "MARKET",
    "SECTOR",
)


def normalize_yfinance_ticker(raw: object) -> str | None:
    """Return upper-case Yahoo symbol or ``None`` if clearly not a tradable ticker."""
    s = str(raw or "").strip().upper()
    if not s:
        return None
    s = s.replace(" ", "").replace("/", "-")
    if "_" in s:
        return None
    if len(s) > 15 or len(s) < 1:
        return None
    for bad in _JUNK_SUBSTR:
        if bad in s:
            return None
    if not _TICKER_OK.match(s):
        return None
    return s
