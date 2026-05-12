"""MCP servers exposed to Agno agents. Singletons are instantiated at import time."""

from src.mcp.report_server import ReportMCPServer
from src.mcp.web_search_server import WebSearchMCPServer
from src.mcp.yfinance_server import YFinanceMCPServer

yfinance_mcp = YFinanceMCPServer()
web_search_mcp = WebSearchMCPServer()
report_mcp = ReportMCPServer()

__all__ = [
    "YFinanceMCPServer",
    "WebSearchMCPServer",
    "ReportMCPServer",
    "yfinance_mcp",
    "web_search_mcp",
    "report_mcp",
]
