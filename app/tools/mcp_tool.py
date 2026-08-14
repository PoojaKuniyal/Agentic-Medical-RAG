"""
Future PubMed MCP server integration.

This is a stub implementation that satisfies the AbstractPubMedTool interface.
When a PubMed MCP server becomes available:
  1. Implement the search() method using the MCP client SDK.
  2. Set PUBMED_TOOL=mcp in .env.
  3. No agent code changes are required.
"""

from __future__ import annotations

import logging

from app.graph.state import PubMedArticle
from app.tools.base import AbstractPubMedTool

logger = logging.getLogger(__name__)


class MCPPubMedTool(AbstractPubMedTool):
    """
    Stub PubMed tool for a future MCP server integration.

    Replace the search() method body with the actual MCP client call
    when the MCP server is available.
    """

    def search(self, query: str, max_results: int | None = None) -> list[PubMedArticle]:
        raise NotImplementedError(
            "MCPPubMedTool is a stub. "
            "Implement search() using your MCP client SDK and set PUBMED_TOOL=mcp."
        )
