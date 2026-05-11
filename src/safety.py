"""
Regex-only safety guard (no LLM). Block regex per category + optional allow-regex for
educational phrasing; first block without a matching allow wins. Patterns compile once in ``SafetyGuard.__init__``.
"""

from __future__ import annotations

import re
from typing import Pattern

from src.models import SafetyVerdict

# Stable refusal copy per policy bucket (shown verbatim when blocked).
_MESSAGES: dict[str, str] = {
    "insider_trading": (
        "I'm unable to assist with trades based on material non-public information. "
        "Trading on insider information is illegal and carries severe criminal and civil penalties. "
        "I'd be happy to discuss publicly available analysis instead."
    ),
    "market_manipulation": (
        "I can't help with schemes designed to artificially influence market prices. "
        "Market manipulation is a serious securities violation. "
        "I can help you understand legitimate trading strategies."
    ),
    "money_laundering": (
        "I'm not able to assist with structuring transactions to avoid regulatory reporting "
        "or conceal the origin of funds. These activities violate AML laws. "
        "I can explain legitimate tax and compliance planning."
    ),
    "guaranteed_returns": (
        "No legitimate investment can guarantee returns. I won't make such representations "
        "as they're both false and potentially fraudulent. "
        "I can help you understand realistic risk-adjusted return expectations."
    ),
    "reckless_advice": (
        "I can't recommend this strategy — it poses severe financial risk to your core financial security. "
        "I'm here to help you build and protect wealth responsibly. "
        "Let's discuss a more balanced approach."
    ),
    "sanctions_evasion": (
        "I'm unable to assist with circumventing OFAC sanctions or other financial regulations. "
        "Sanctions evasion carries criminal penalties. "
        "I can explain how sanctions screening works."
    ),
    "fraud": (
        "I can't help create false or misleading financial documents. "
        "This constitutes fraud and carries criminal liability. "
        "I'm happy to help with legitimate financial planning."
    ),
}


class SafetyGuard:
    """Allow-regexes deliberately omit bare ``what's …`` so e.g. \"what's a foolproof way\" stays blocked under guaranteed_returns."""

    def __init__(self) -> None:
        self._rules: list[tuple[str, Pattern[str], Pattern[str] | None]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        _EDU_INSIDER = re.compile(
            r"(?ix)\b("
            r"what\s+is(\s+the)?|what\s+are(\s+the)?|explain(\s+the)?|"
            r"how\s+does|how\s+do\s+regulators|"
            r"what'?s\s+the\s+difference|difference\s+between|"
            r"penalt(y|ies)|disclosure\s+rules|investigate|regulations?\b"
            r")\b"
        )

        _EDU_MANIP = re.compile(
            r"(?ix)\b("
            r"what\s+is(\s+a)?|explain|how\s+does|how\s+do\s+regulators|"
            r"are\b.*\blegal\b|why\s+is\b.*\billegal\b"
            r")\b"
        )

        _EDU_AML = re.compile(
            r"(?ix)\b("
            r"what\s+are(\s+the)?|what\s+is|describe(\s+the)?|"
            r"three\s+stages\s+of\s+money\s+laundering|"
            r"reporting\s+requirements|financial\s+regulation"
            r")\b"
        )

        _EDU_GUARANTEED = re.compile(
            r"(?ix)\b("
            r"what\s+is|why\s+are|explain|is\s+it\s+ever\s+legitimate|"
            r"historical\s+average|average\s+annual\s+return|red\s+flag|ponzi"
            r")\b"
        )

        _EDU_RECKLESS = re.compile(
            r"(?ix)\b("
            r"what\s+factors|what\s+are\s+the\s+risks|how\s+should"
            r")\b"
        )

        _EDU_SANCTIONS = re.compile(
            r"(?ix)\b("
            r"what\s+are\s+ofac|how\s+do\s+brokerages|compliance\s+obligations"
            r")\b"
        )

        _insider_block = re.compile(
            r"(?ix)(?:"
            r"\b(?:i\s+)?work\s+at\b.*\b(?:know|knew)\b.*\bunannounced\b.*\b(?:buy|shares)\b"
            r"|"
            r"\bfriend\s+at\b.*\b(?:tip|tipped)\b.*\b(?:earnings)\b.*\b(?:load\s+up|before)\b"
            r"|"
            r"\bhelp\s+me\s+trade\b.*\b(?:confidential|merger)\b"
            r"|"
            r"\bneed\s+to\s+know\s+the\s+earnings\s+before\b.*\bannouncement\b"
            r")"
        )

        _manip_block = re.compile(
            r"(?ix)(?:"
            r"\bpump\b.*\bprice\b.*\b(?:before\s+selling|dump)\b"
            r"|"
            r"\bcoordinated\s+buying\s+scheme\b.*\bmove\b"
            r"|"
            r"\bhelp\s+me\s+wash\s+trade\b"
            r")"
        )

        _ml_block = re.compile(
            r"(?ix)(?:"
            r"\bmove\b.*\bcash\b.*\bstock\b.*\bwithout\s+reporting\b"
            r"|"
            r"\bhelp\s+me\s+structure\b.*\b(?:avoid|10k|threshold)\b"
            r"|"
            r"\blayer\s+my\s+trades\b.*\bobscure\b"
            r"|"
            r"\bhelp\s+me\s+hide\b.*\b(?:profits|trading)\b.*\b(?:tax|authorities)\b"
            r")"
        )

        _guar_block = re.compile(
            r"(?ix)(?:"
            r"\bguarantee\s+me\b"
            r"|"
            r"\bpromise\s+me\b.*\b(?:double|returns)\b"
            r"|"
            r"\b100%\s+certain\b.*\bgo\s+up\b"
            r"|"
            r"\bfoolproof\b.*\b(?:make|month|returns|way)\b"
            r")"
        )

        _reck_block = re.compile(
            r"(?ix)(?:"
            r"\b70\b.*\bretirement\b.*\bcrypto\b.*\bdo\s+it\b"
            r"|"
            r"\btell\s+me\s+to\s+take\s+a\s+margin\s+loan\b"
            r"|"
            r"\bentire\s+emergency\s+fund\b.*\boptions\b"
            r"|"
            r"\bmortgage\s+my\s+house\b.*\b(?:stock|for)\b"
            r")"
        )

        _san_block = re.compile(
            r"(?ix)(?:"
            r"\bhelp\s+me\s+route\s+trades\b.*\b(?:shell\s+company|bypass)\b.*\bofac\b"
            r"|"
            r"\bhow\s+do\s+i\s+invest\b.*\bsanctioned\b.*\b(?:without|traced)\b"
            r")"
        )

        _fraud_block = re.compile(
            r"(?ix)\b(?:draft|create|fabricate)\b.*\bfake\b.*\bcontract\s+note\b"
        )

        self._rules = [
            ("insider_trading", _insider_block, _EDU_INSIDER),
            ("market_manipulation", _manip_block, _EDU_MANIP),
            ("money_laundering", _ml_block, _EDU_AML),
            ("guaranteed_returns", _guar_block, _EDU_GUARANTEED),
            ("reckless_advice", _reck_block, _EDU_RECKLESS),
            ("sanctions_evasion", _san_block, _EDU_SANCTIONS),
            ("fraud", _fraud_block, None),
        ]

    def check(self, query: str) -> SafetyVerdict:
        text = query.strip()
        lowered = text.lower()

        for category, block_pat, allow_pat in self._rules:
            if not block_pat.search(lowered):
                continue
            if allow_pat is not None and allow_pat.search(lowered):
                continue
            return SafetyVerdict(
                blocked=True,
                category=category,
                message=_MESSAGES[category],
            )

        return SafetyVerdict(blocked=False, category="clean", message="")


_guard = SafetyGuard()


def check(query: str) -> SafetyVerdict:
    return _guard.check(query)
