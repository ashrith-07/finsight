"""Report generation MCP toolkit. Produces Markdown or PDF artefacts under ``reports/``."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agno.tools import Toolkit
from fpdf import FPDF

from src.logging_config import get_logger

logger = get_logger("mcp.report")

DISCLAIMER = (
    "This report is for informational purposes only and does not constitute "
    "investment advice. All data sourced from public market feeds. "
    "Please consult a qualified financial adviser before acting."
)


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_").lower()
    return s or "report"


def _fmt_money(v, currency: str = "USD") -> str:
    try:
        return f"{currency} {float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v) -> str:
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


# fpdf core fonts are Latin-1 only; strip anything outside that range to keep PDF rendering safe.
def _ascii(text: str) -> str:
    return str(text or "").encode("latin-1", "ignore").decode("latin-1")


class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "Valura AI — Report", ln=1)
        self.set_draw_color(180, 180, 180)
        self.line(10, 20, 200, 20)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def _write_pdf(file_path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    pdf = _PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _ascii(title), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z", ln=1)
    pdf.ln(4)

    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _ascii(heading), ln=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _ascii(body))
        pdf.ln(2)

    pdf.output(str(file_path))


class ReportMCPServer(Toolkit):
    """Generates Markdown / PDF reports for portfolio, market and risk workflows."""

    name = "report_mcp"
    OUTPUT_DIR = Path("reports")

    def __init__(self) -> None:
        super().__init__(name="report_mcp", auto_register=False)
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.register(self.generate_portfolio_report)
        self.register(self.generate_market_report)
        self.register(self.generate_risk_report)

    # ---------- Portfolio ----------
    def generate_portfolio_report(
        self,
        user_name: str,
        portfolio_data: dict,
        format: str = "markdown",
    ) -> dict:
        data = portfolio_data or {}
        currency = data.get("base_currency") or data.get("currency") or "USD"
        positions = data.get("positions") or []
        observations = data.get("observations") or []
        concentration = data.get("concentration_risk") or {}

        title = f"Portfolio Health Report — {user_name}"
        summary_lines = [
            f"- Current value: {_fmt_money(data.get('current_value'), currency)}",
            f"- Total return: {_fmt_pct(data.get('total_return_pct'))}",
            f"- Benchmark: {data.get('benchmark') or 'n/a'}",
            f"- Benchmark return: {_fmt_pct(data.get('benchmark_return_pct'))}",
        ]

        top_weight = concentration.get("top_holding_weight_pct")
        try:
            top_weight_str = f"{float(top_weight):.2f}%"
        except (TypeError, ValueError):
            top_weight_str = "—"
        conc_lines = [
            f"- Flag: {concentration.get('flag') or 'n/a'}",
            f"- Top holding weight: {top_weight_str}",
        ]
        if concentration.get("note"):
            conc_lines.append(f"- Note: {concentration['note']}")

        position_lines = ["| Ticker | Quantity | Avg Cost | Currency |", "|---|---:|---:|:---:|"]
        for p in positions:
            position_lines.append(
                f"| {p.get('ticker', '?')} "
                f"| {_fmt_num(p.get('quantity'))} "
                f"| {_fmt_num(p.get('avg_cost'))} "
                f"| {p.get('currency') or currency} |"
            )

        obs_lines = [
            f"- **{o.get('severity', 'info').upper()}**: {o.get('text', '')}"
            for o in observations
        ] or ["- (no observations)"]

        sections = [
            ("Executive Summary", "\n".join(summary_lines)),
            ("Concentration Risk Analysis", "\n".join(conc_lines)),
            ("Position Details", "\n".join(position_lines)),
            ("Key Observations", "\n".join(obs_lines)),
            ("Disclaimer", DISCLAIMER),
        ]

        return self._emit(title, sections, format=format, prefix=f"portfolio_{_slug(user_name)}")

    # ---------- Market ----------
    def generate_market_report(
        self,
        tickers: list[str],
        snapshots: list[dict],
        news: list[dict],
        format: str = "markdown",
    ) -> dict:
        title = f"Market Research Report — {', '.join(tickers) if tickers else 'overview'}"

        snap_lines = [
            "| Ticker | Price | Day % | 52w High | 52w Low | Market Cap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        tech_lines = []
        for s in snapshots or []:
            snap_lines.append(
                f"| {s.get('ticker', '?')} "
                f"| {_fmt_num(s.get('current_price'))} "
                f"| {_fmt_pct(s.get('day_change_pct'))} "
                f"| {_fmt_num(s.get('fifty_two_week_high'))} "
                f"| {_fmt_num(s.get('fifty_two_week_low'))} "
                f"| {_fmt_num(s.get('market_cap'))} |"
            )
            tech_lines.append(
                f"- {s.get('ticker', '?')}: current {_fmt_num(s.get('current_price'))}, "
                f"52w high {_fmt_num(s.get('fifty_two_week_high'))}, "
                f"52w low {_fmt_num(s.get('fifty_two_week_low'))}"
            )

        news_lines = []
        for n in (news or [])[:10]:
            title_str = n.get("title") or "(untitled)"
            url = n.get("url") or ""
            date = n.get("published_date") or ""
            news_lines.append(f"- [{title_str}]({url}) — {date}")
        if not news_lines:
            news_lines = ["- (no news available)"]

        sections = [
            ("Market Snapshot", "\n".join(snap_lines)),
            ("Recent News", "\n".join(news_lines)),
            ("Technical Levels", "\n".join(tech_lines) or "- (no snapshots)"),
            ("Disclaimer", DISCLAIMER),
        ]

        prefix = f"market_{_slug('_'.join(tickers))}" if tickers else "market_overview"
        return self._emit(title, sections, format=format, prefix=prefix)

    # ---------- Risk ----------
    def generate_risk_report(self, risk_data: dict, format: str = "markdown") -> dict:
        data = risk_data or {}
        title = "Risk Analysis Report"

        summary_lines = [
            f"- Overall risk score: {data.get('risk_score', 'n/a')}",
            f"- Profile: {data.get('risk_profile', 'n/a')}",
            f"- Time horizon: {data.get('time_horizon', 'n/a')}",
        ]

        var_data = data.get("var") or {}
        var_lines = [
            f"- 1-day 95% VaR: {_fmt_money(var_data.get('one_day_95'), data.get('currency', 'USD'))}",
            f"- 1-day 99% VaR: {_fmt_money(var_data.get('one_day_99'), data.get('currency', 'USD'))}",
            f"- Method: {var_data.get('method', 'historical simulation')}",
        ]

        stress_lines = [
            f"- {s.get('scenario', '?')}: {_fmt_pct(s.get('impact_pct'))}"
            for s in (data.get("stress_tests") or [])
        ] or ["- (no stress tests run)"]

        rec_lines = [f"- {r}" for r in (data.get("recommendations") or [])] or [
            "- (no recommendations)"
        ]

        sections = [
            ("Risk Summary", "\n".join(summary_lines)),
            ("VaR Analysis", "\n".join(var_lines)),
            ("Stress Test Results", "\n".join(stress_lines)),
            ("Recommendations", "\n".join(rec_lines)),
            ("Disclaimer", DISCLAIMER),
        ]

        return self._emit(title, sections, format=format, prefix="risk")

    # ---------- internals ----------
    def _emit(
        self,
        title: str,
        sections: list[tuple[str, str]],
        *,
        format: str,
        prefix: str,
    ) -> dict:
        fmt = (format or "markdown").lower()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        if fmt == "pdf":
            filename = f"{prefix}_{ts}.pdf"
            file_path = self.OUTPUT_DIR / filename
            try:
                _write_pdf(file_path, title, sections)
            except Exception as e:
                logger.error("report_mcp PDF write failed (%s): %s", file_path, e)
                return {
                    "format": "pdf",
                    "filename": filename,
                    "file_path": str(file_path),
                    "generated_at": generated_at,
                    "error": str(e),
                }
            return {
                "format": "pdf",
                "filename": filename,
                "file_path": str(file_path),
                "generated_at": generated_at,
            }

        # markdown (default)
        md_parts = [f"# {title}", f"_Generated: {generated_at}_", ""]
        for heading, body in sections:
            md_parts.append(f"## {heading}")
            md_parts.append(body)
            md_parts.append("")
        content = "\n".join(md_parts).rstrip() + "\n"

        filename = f"{prefix}_{ts}.md"
        file_path = self.OUTPUT_DIR / filename
        try:
            file_path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.error("report_mcp markdown write failed (%s): %s", file_path, e)
            return {
                "format": "markdown",
                "filename": filename,
                "content": content,
                "file_path": str(file_path),
                "generated_at": generated_at,
                "error": str(e),
            }

        return {
            "format": "markdown",
            "filename": filename,
            "content": content,
            "file_path": str(file_path),
            "generated_at": generated_at,
        }
