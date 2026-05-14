"""Pure-math MCP toolkit: compound interest, DCA, Black-Scholes, loans, retirement, rebalancing.

All tools are deterministic and do no network I/O — safe to run in any worker
thread, no rate limits, no retries needed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from agno.tools import Toolkit
from scipy.stats import norm

from src.logging_config import get_logger

logger = get_logger("mcp.calculator")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or math.isinf(f):
        return default
    return f


class CalculatorMCPServer(Toolkit):
    """Financial calculations exposed as Agno tools."""

    name = "calculator_mcp"

    def __init__(self) -> None:
        super().__init__(name="calculator_mcp", auto_register=False)
        self.register(self.compound_interest)
        self.register(self.dca_projection)
        self.register(self.options_black_scholes)
        self.register(self.loan_amortisation)
        self.register(self.retirement_projection)
        self.register(self.portfolio_rebalance_trades)

    def compound_interest(
        self,
        principal: float,
        annual_rate: float,
        years: int,
        compounds_per_year: int = 12,
        monthly_contribution: float = 0.0,
    ) -> dict:
        """Compound growth + monthly contributions; returns final value, totals,
        and a year-by-year breakdown.

        Closed-form: A = P(1+r/n)^(nt) + PMT_n * ((1+r/n)^(nt)-1) / (r/n).
        Monthly PMT is converted to per-period PMT to match ``compounds_per_year``.
        """
        try:
            P = _safe_float(principal)
            r = _safe_float(annual_rate)
            n = max(int(compounds_per_year or 12), 1)
            t = max(int(years or 0), 0)
            pmt_monthly = _safe_float(monthly_contribution)

            per_period_rate = r / n
            pmt_per_period = pmt_monthly * 12 / n

            year_breakdown: list[dict] = [{"year": 0, "value": round(P, 2)}]
            value = P
            for year in range(1, t + 1):
                for _ in range(n):
                    value = value * (1 + per_period_rate) + pmt_per_period
                year_breakdown.append({"year": year, "value": round(value, 2)})

            total_contributed = round(P + pmt_monthly * 12 * t, 2)
            return {
                "final_value": round(value, 2),
                "total_contributed": total_contributed,
                "total_interest_earned": round(value - total_contributed, 2),
                "principal": round(P, 2),
                "annual_rate": r,
                "years": t,
                "compounds_per_year": n,
                "monthly_contribution": pmt_monthly,
                "year_by_year": year_breakdown,
            }
        except Exception as e:
            logger.warning("calculator_mcp.compound_interest failed: %s", e)
            return {"error": str(e)}

    def dca_projection(
        self,
        monthly_amount: float,
        annual_return_pct: float,
        years: int,
        initial_investment: float = 0.0,
    ) -> dict:
        """Dollar-cost-averaging projection with month-by-month value."""
        try:
            pmt = _safe_float(monthly_amount)
            r_annual = _safe_float(annual_return_pct) / 100.0
            t = max(int(years or 0), 0)
            P = _safe_float(initial_investment)
            r_monthly = r_annual / 12.0
            months = t * 12

            value = P
            invested = P
            breakdown: list[dict] = []
            for m in range(1, months + 1):
                value = value * (1 + r_monthly) + pmt
                invested += pmt
                # Sample monthly for first year, then quarterly to keep payload sane.
                if m <= 12 or m % 3 == 0 or m == months:
                    breakdown.append({
                        "month": m,
                        "invested": round(invested, 2),
                        "value": round(value, 2),
                    })

            gain = value - invested
            return {
                "final_value": round(value, 2),
                "total_invested": round(invested, 2),
                "total_gain": round(gain, 2),
                "gain_pct": round((gain / invested * 100) if invested > 0 else 0.0, 2),
                "monthly_amount": pmt,
                "initial_investment": round(P, 2),
                "annual_return_pct": _safe_float(annual_return_pct),
                "years": t,
                "monthly_breakdown": breakdown,
            }
        except Exception as e:
            logger.warning("calculator_mcp.dca_projection failed: %s", e)
            return {"error": str(e)}

    def options_black_scholes(
        self,
        stock_price: float,
        strike_price: float,
        time_to_expiry_days: int,
        volatility_pct: float,
        risk_free_rate: float = 0.0525,
        option_type: str = "call",
    ) -> dict:
        """Black-Scholes price + Greeks for European calls/puts."""
        try:
            S = _safe_float(stock_price)
            K = _safe_float(strike_price)
            T = max(int(time_to_expiry_days or 0), 1) / 365.0
            sigma = _safe_float(volatility_pct) / 100.0
            r = _safe_float(risk_free_rate)
            kind = (option_type or "call").lower().strip()

            if S <= 0 or K <= 0 or sigma <= 0:
                return {"error": "stock_price, strike_price, and volatility must be > 0"}

            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            N = norm.cdf
            n = norm.pdf

            if kind == "put":
                price = K * math.exp(-r * T) * N(-d2) - S * N(-d1)
                delta = N(d1) - 1.0
                rho = -K * T * math.exp(-r * T) * N(-d2)
                # Theta in option-days (divide annual by 365 for "per day" decay).
                theta_annual = -(S * n(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * N(-d2)
            else:
                kind = "call"
                price = S * N(d1) - K * math.exp(-r * T) * N(d2)
                delta = N(d1)
                rho = K * T * math.exp(-r * T) * N(d2)
                theta_annual = -(S * n(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N(d2)

            gamma = n(d1) / (S * sigma * math.sqrt(T))
            vega = S * n(d1) * math.sqrt(T)

            return {
                "option_type": kind,
                "option_price": round(float(price), 4),
                "delta": round(float(delta), 4),
                "gamma": round(float(gamma), 6),
                "theta_per_day": round(float(theta_annual) / 365.0, 4),
                "vega_per_1pct": round(float(vega) / 100.0, 4),
                "rho_per_1pct": round(float(rho) / 100.0, 4),
                "stock_price": S,
                "strike_price": K,
                "time_to_expiry_days": int(time_to_expiry_days or 0),
                "volatility_pct": _safe_float(volatility_pct),
                "risk_free_rate": r,
            }
        except Exception as e:
            logger.warning("calculator_mcp.options_black_scholes failed: %s", e)
            return {"error": str(e)}

    def loan_amortisation(
        self,
        principal: float,
        annual_rate: float,
        years: int,
    ) -> dict:
        """Standard fixed-rate monthly amortisation schedule."""
        try:
            P = _safe_float(principal)
            r_annual = _safe_float(annual_rate)
            t = max(int(years or 0), 0)
            n = t * 12
            r = r_annual / 12.0

            if P <= 0 or n <= 0:
                return {"error": "principal and years must be > 0"}

            if r == 0:
                monthly_payment = P / n
            else:
                monthly_payment = P * r * (1 + r) ** n / ((1 + r) ** n - 1)

            balance = P
            first_year: list[dict] = []
            total_interest = 0.0
            for m in range(1, n + 1):
                interest = balance * r
                principal_paid = monthly_payment - interest
                balance = max(balance - principal_paid, 0.0)
                total_interest += interest
                if m <= 12:
                    first_year.append({
                        "month": m,
                        "payment": round(monthly_payment, 2),
                        "principal": round(principal_paid, 2),
                        "interest": round(interest, 2),
                        "balance": round(balance, 2),
                    })

            payoff_date = (date.today() + timedelta(days=int(n * 30.4375))).isoformat()
            total_paid = monthly_payment * n
            return {
                "monthly_payment": round(monthly_payment, 2),
                "total_paid": round(total_paid, 2),
                "total_interest": round(total_interest, 2),
                "principal": round(P, 2),
                "annual_rate": r_annual,
                "years": t,
                "payoff_date": payoff_date,
                "first_year_breakdown": first_year,
            }
        except Exception as e:
            logger.warning("calculator_mcp.loan_amortisation failed: %s", e)
            return {"error": str(e)}

    def retirement_projection(
        self,
        current_age: int,
        retirement_age: int,
        current_savings: float,
        monthly_contribution: float,
        expected_annual_return: float,
        monthly_expenses_in_retirement: float,
    ) -> dict:
        """Retirement readiness check using compound growth + the 4% rule."""
        try:
            ca = max(int(current_age or 0), 0)
            ra = max(int(retirement_age or 0), ca)
            years_to_retire = ra - ca

            grow = self.compound_interest(
                principal=current_savings,
                annual_rate=expected_annual_return,
                years=years_to_retire,
                compounds_per_year=12,
                monthly_contribution=monthly_contribution,
            )
            if "error" in grow:
                return grow

            projected = grow["final_value"]
            monthly_expenses = _safe_float(monthly_expenses_in_retirement)
            annual_expenses = monthly_expenses * 12

            # 4% rule: sustainable annual income ≈ 4% of nest egg.
            sustainable_monthly = projected * 0.04 / 12.0
            shortfall = sustainable_monthly - monthly_expenses
            years_savings_will_last = (
                projected / annual_expenses if annual_expenses > 0 else float("inf")
            )

            # Solve for required monthly contribution to exactly meet the goal:
            # FV_target = monthly_expenses * 12 / 0.04  (4%-rule capital).
            target_capital = (annual_expenses / 0.04) if annual_expenses > 0 else projected
            r_monthly = _safe_float(expected_annual_return) / 12.0
            n_months = years_to_retire * 12
            P = _safe_float(current_savings)

            if n_months <= 0:
                recommended_pmt = 0.0
            elif r_monthly == 0:
                recommended_pmt = max((target_capital - P) / n_months, 0.0)
            else:
                growth_factor = (1 + r_monthly) ** n_months
                fv_principal = P * growth_factor
                annuity_factor = (growth_factor - 1) / r_monthly
                needed = max(target_capital - fv_principal, 0.0)
                recommended_pmt = needed / annuity_factor if annuity_factor > 0 else 0.0

            return {
                "current_age": ca,
                "retirement_age": ra,
                "years_to_retirement": years_to_retire,
                "projected_savings_at_retirement": projected,
                "target_capital_4pct_rule": round(target_capital, 2),
                "monthly_income_sustainable_forever": round(sustainable_monthly, 2),
                "monthly_expenses_in_retirement": round(monthly_expenses, 2),
                "shortfall_or_surplus": round(shortfall, 2),
                "years_savings_will_last": (
                    round(years_savings_will_last, 1)
                    if math.isfinite(years_savings_will_last) else None
                ),
                "recommended_monthly_contribution_to_hit_goal": round(recommended_pmt, 2),
                "current_monthly_contribution": _safe_float(monthly_contribution),
                "expected_annual_return": _safe_float(expected_annual_return),
            }
        except Exception as e:
            logger.warning("calculator_mcp.retirement_projection failed: %s", e)
            return {"error": str(e)}

    def portfolio_rebalance_trades(
        self,
        current_positions: list[dict],
        target_allocations: dict,
        total_value: float,
    ) -> dict:
        """Translate target weights into concrete buy/sell orders.

        ``current_positions`` rows: {ticker, current_value, current_pct, [price]}.
        ``target_allocations``: {ticker: target_pct} (percent, not fraction).
        """
        try:
            total = _safe_float(total_value)
            if total <= 0:
                return {"error": "total_value must be > 0"}

            by_ticker: dict[str, dict] = {}
            for row in current_positions or []:
                if not isinstance(row, dict):
                    continue
                t = str(row.get("ticker") or "").upper()
                if not t:
                    continue
                cv = _safe_float(row.get("current_value"))
                price = _safe_float(row.get("price"))
                cpct = _safe_float(row.get("current_pct")) or (cv / total * 100 if total > 0 else 0.0)
                by_ticker[t] = {
                    "current_value": cv,
                    "price": price,
                    "current_pct": cpct,
                }

            trades: list[dict] = []
            all_tickers = set(by_ticker) | {str(k).upper() for k in (target_allocations or {})}
            for t in sorted(all_tickers):
                target_pct = _safe_float((target_allocations or {}).get(t)) or _safe_float(
                    (target_allocations or {}).get(t.lower())
                )
                cur = by_ticker.get(t, {"current_value": 0.0, "price": 0.0, "current_pct": 0.0})
                target_value = total * target_pct / 100.0
                delta_value = target_value - cur["current_value"]
                action = "buy" if delta_value > 0 else ("sell" if delta_value < 0 else "hold")
                shares = (
                    round(delta_value / cur["price"], 4)
                    if cur["price"] and cur["price"] > 0 else None
                )
                trades.append({
                    "ticker": t,
                    "action": action,
                    "shares": shares,
                    "dollar_amount": round(abs(delta_value), 2),
                    "current_pct": round(cur["current_pct"], 2),
                    "target_pct": round(target_pct, 2),
                    "current_value": round(cur["current_value"], 2),
                    "target_value": round(target_value, 2),
                })

            actionable = [t for t in trades if t["action"] != "hold" and t["dollar_amount"] >= 1.0]
            return {
                "total_value": round(total, 2),
                "trades": trades,
                "actionable_trades": actionable,
                "total_trade_value": round(sum(t["dollar_amount"] for t in actionable), 2),
            }
        except Exception as e:
            logger.warning("calculator_mcp.portfolio_rebalance_trades failed: %s", e)
            return {"error": str(e)}
