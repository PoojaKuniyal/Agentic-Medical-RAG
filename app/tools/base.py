"""
Abstract base class for PubMed tool implementations.

Design intent
─────────────
The PubMed Agent always calls AbstractPubMedTool.search() — it has no
knowledge of whether the underlying transport is NCBI Entrez or a PubMed MCP
server.

To swap implementations:
  1. Implement a new subclass of AbstractPubMedTool.
  2. Update PUBMED_TOOL in .env (entrez | mcp).
  3. Update get_pubmed_tool() in this module.
  No other code changes are required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.graph.state import PubMedArticle


class AbstractPubMedTool(ABC):
    """Interface contract for all PubMed tool implementations."""

    @abstractmethod
    def search(self, query: str, max_results: int | None = None) -> list[PubMedArticle]:
        """
        Search PubMed and return a list of article records.

        Parameters
        ----------
        query : str
            The biomedical search query.
        max_results : int | None
            Maximum number of results to return.
            If None, uses the default from settings.

        Returns
        -------
        list[PubMedArticle]
            Structured article records — never raw API responses.
        """
        ...


def get_pubmed_tool() -> AbstractPubMedTool:
    """
    Factory function that returns the configured PubMed tool implementation.

    Reads PUBMED_TOOL from settings:
      • entrez — EntrezPubMedTool (NCBI Entrez REST API)
      • mcp    — MCPPubMedTool (future MCP server integration)
    """
    from app.config import get_settings

    settings = get_settings()
    tool_type = settings.pubmed_tool

    if tool_type == "entrez":
        from app.tools.entrez_tool import EntrezPubMedTool

        return EntrezPubMedTool()

    elif tool_type == "mcp":
        from app.tools.mcp_tool import MCPPubMedTool

        return MCPPubMedTool()

    else:
        raise ValueError(
            f"Unsupported PUBMED_TOOL='{tool_type}'. Supported: entrez | mcp"
        )
