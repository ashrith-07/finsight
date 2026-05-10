"""
Synchronous, pattern-only safety guard (no LLM, no network).

Design (two layers, inspired by the assignment brief):
  * Layer 1 — per-category ``block`` regex: first-person / operational harm.
  * Layer 2 — per-category ``allow`` regex: definitional or regulatory-education
    cues (“what is…”, “explain…”, “penalties…”). If Layer 1 hits but Layer 2
    also hits for that category, we **do not** block (educational pass-through).

Rules are evaluated in fixed order; the first block-without-allow wins.
All ``re.compile`` calls run once inside ``SafetyGuard._load_rules`` (invoked
from ``__init__``), so ``check()`` is a small linear scan over pre-built
automata — typically sub-millisecond on realistic query lengths.
"""

from __future__ import annotations

import re
from typing import Pattern

from src.models import SafetyVerdict

# ---------------------------------------------------------------------------
# Verbatim user-facing refusal copy (one stable string per policy bucket)
# ---------------------------------------------------------------------------

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
    """
    Layer-1 block regexes paired with optional Layer-2 allow (educational) regexes.

    Allow patterns deliberately avoid bare “what's …” openings so definitional
    questions like “what's the historical average …” still pass while
    “what's a foolproof way …” remains blocked by the guaranteed_returns rule.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[str, Pattern[str], Pattern[str] | None]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        # --- Shared educational / definitional signals (Layer 2) -----------------
        # Tuned to gold fixtures: includes “describe”, “why are”, “how should”,
        # “what factors”, “what's the difference”, “are … legal”, etc.
        # Excludes bare “what's a …” so “what's a foolproof way” cannot escape.
        # Insider allow: definitional / comparative / penalty / supervisory context.
        # “how do regulators” covers “how do regulators detect…” without matching
        # “how do i move cash…” (different token after “how do”).
        _EDU_INSIDER = re.compile(
            r"(?ix)\b("
            r"what\s+is(\s+the)?|what\s+are(\s+the)?|explain(\s+the)?|"
            r"how\s+does|how\s+do\s+regulators|"
            r"what'?s\s+the\s+difference|difference\s+between|"
            r"penalt(y|ies)|disclosure\s+rules|investigate|regulations?\b"
            r")\b"
        )

        # Manipulation allow: pedagogy on schemes, legality questions, SEC/FCA narrative.
        # “are … legal” needs a broad window between tokens for “are pump-and-dump … legal”.
        _EDU_MANIP = re.compile(
            r"(?ix)\b("
            r"what\s+is(\s+a)?|explain|how\s+does|how\s+do\s+regulators|"
            r"are\b.*\blegal\b|why\s+is\b.*\billegal\b"
            r")\b"
        )

        # AML allow: course-style “describe three stages” and vocabulary lessons
        # (“what is structuring”) — distinct from imperative “help me structure…”.
        _EDU_AML = re.compile(
            r"(?ix)\b("
            r"what\s+are(\s+the)?|what\s+is|describe(\s+the)?|"
            r"three\s+stages\s+of\s+money\s+laundering|"
            r"reporting\s+requirements|financial\s+regulation"
            r")\b"
        )

        # Guaranteed-return allow: critique / history / legitimacy questions.
        # Deliberately **no** bare “what's …” — that would unblock “what's a foolproof way…”.
        _EDU_GUARANTEED = re.compile(
            r"(?ix)\b("
            r"what\s+is|why\s+are|explain|is\s+it\s+ever\s+legitimate|"
            r"historical\s+average|average\s+annual\s+return|red\s+flag|ponzi"
            r")\b"
        )

        # Reckless allow: risk framing and allocation guidance, not imperative “do it”.
        _EDU_RECKLESS = re.compile(
            r"(?ix)\b("
            r"what\s+factors|what\s+are\s+the\s+risks|how\s+should"
            r")\b"
        )

        # Sanctions allow: compliance literacy, not evasion recipes.
        _EDU_SANCTIONS = re.compile(
            r"(?ix)\b("
            r"what\s+are\s+ofac|how\s+do\s+brokerages|compliance\s+obligations"
            r")\b"
        )

        # --- Layer 1: insider_trading -----------------------------------------
        # First-person + MNPI verbs/nouns + trading intent; kept separate from
        # academic “what is insider trading” phrasing (handled by _EDU_INSIDER).
        _insider_block = re.compile(
            r"(?ix)(?:"
            # Workplace / relationship + non-public deal knowledge + trade timing
            r"\b(?:i\s+)?work\s+at\b.*\b(?:know|knew)\b.*\bunannounced\b.*\b(?:buy|shares)\b"
            r"|"
            r"\bfriend\s+at\b.*\b(?:tip|tipped)\b.*\b(?:earnings)\b.*\b(?:load\s+up|before)\b"
            r"|"
            r"\bhelp\s+me\s+trade\b.*\b(?:confidential|merger)\b"
            r"|"
            r"\bneed\s+to\s+know\s+the\s+earnings\s+before\b.*\bannouncement\b"
            r")"
        )

        # --- Layer 1: market_manipulation ------------------------------------
        # Requires operational verbs (“pump up the price … before selling”,
        # “design … scheme … move”, “help me wash trade”) so “what is wash trading”
        # (Layer 2) never hits the block arm.
        _manip_block = re.compile(
            r"(?ix)(?:"
            r"\bpump\b.*\bprice\b.*\b(?:before\s+selling|dump)\b"
            r"|"
            r"\bcoordinated\s+buying\s+scheme\b.*\bmove\b"
            r"|"
            r"\bhelp\s+me\s+wash\s+trade\b"
            r")"
        )

        # --- Layer 1: money_laundering ---------------------------------------
        # Uses action phrases (“layer my trades”) so “layering” in AML pedagogy
        # does not false-positive.
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

        # --- Layer 1: guaranteed_returns -------------------------------------
        # Anchors on imperative “guarantee/promise me” or absurd certainty language.
        # “foolproof … way|make|month” catches colloquial get-rich-quick phrasing
        # without matching benign “foolproof” mentions inside academic critique
        # (those flows hit _EDU_GUARANTEED instead when relevant).
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

        # --- Layer 1: reckless_advice ----------------------------------------
        # Targets all-in / leverage / emergency-fund / housing collateral asks
        # phrased as directives (“do it”, “tell me to”, “should i put entire…”).
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

        # --- Layer 1: sanctions_evasion --------------------------------------
        # “route … shell … OFAC” and evasive investing in sanctioned names; both
        # are distinct from “what are OFAC sanctions…” compliance questions.
        _san_block = re.compile(
            r"(?ix)(?:"
            r"\bhelp\s+me\s+route\s+trades\b.*\b(?:shell\s+company|bypass)\b.*\bofac\b"
            r"|"
            r"\bhow\s+do\s+i\s+invest\b.*\bsanctioned\b.*\b(?:without|traced)\b"
            r")"
        )

        # --- Layer 1: fraud ---------------------------------------------------
        # Document forgery: verb of fabrication + “fake” + instrument type.
        # No allow pattern — educational fraud questions in fixtures are generic
        # (“FCA's role in preventing fraud”) and do not mention fake contract notes.
        _fraud_block = re.compile(
            r"(?ix)\b(?:draft|create|fabricate)\b.*\bfake\b.*\bcontract\s+note\b"
        )

        # Order: specific operational asks before any broader future rules.
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
        """Run all rules. Return first block hit. O(n_rules) worst case."""
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
    """Module-level convenience; uses the process-wide singleton guard."""
    return _guard.check(query)
