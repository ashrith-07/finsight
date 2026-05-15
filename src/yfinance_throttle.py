"""Serialize Yahoo Finance access from shared IPs (e.g. Hugging Face Spaces).

Parallel ``ThreadPoolExecutor`` calls to yfinance trigger 429 / Invalid Crumb fast.
Calls are spaced with a small minimum interval under a process-wide lock.
"""

from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_last_mono = 0.0


def yfinance_pause() -> None:
    """Block until at least ``YFINANCE_MIN_INTERVAL_S`` since the last yfinance call."""
    interval = max(0.05, float(os.environ.get("YFINANCE_MIN_INTERVAL_S", "0.55")))
    global _last_mono
    with _lock:
        now = time.monotonic()
        wait = interval - (now - _last_mono)
        if wait > 0:
            time.sleep(wait)
        _last_mono = time.monotonic()
