"""MCP servers exposed to Agno agents. Singletons are instantiated at import time."""

from src.mcp.calculator_server import CalculatorMCPServer
from src.mcp.portfolio_analytics_server import PortfolioAnalyticsMCPServer
from src.mcp.report_server import ReportMCPServer
from src.mcp.web_search_server import WebSearchMCPServer
from src.mcp.yfinance_server import YFinanceMCPServer

yfinance_mcp = YFinanceMCPServer()
web_search_mcp = WebSearchMCPServer()
report_mcp = ReportMCPServer()
calculator_mcp = CalculatorMCPServer()
portfolio_analytics_mcp = PortfolioAnalyticsMCPServer()

__all__ = [
    "YFinanceMCPServer",
    "WebSearchMCPServer",
    "ReportMCPServer",
    "CalculatorMCPServer",
    "PortfolioAnalyticsMCPServer",
    "yfinance_mcp",
    "web_search_mcp",
    "report_mcp",
    "calculator_mcp",
    "portfolio_analytics_mcp",
]
