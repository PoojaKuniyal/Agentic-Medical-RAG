"""
Centralised configuration using Pydantic BaseSettings.
This file centralizes configuration, validates it, and makes it available consistently across the application.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["google_genai", "openai", "anthropic", "groq", "ollama", "open_source"] = "ollama"
    llm_model: str = "llama3"
    llm_fast_provider: Optional[Literal["google_genai", "openai", "anthropic", "groq", "ollama", "open_source"]] = Field(default=None, alias="LLM_FAST_PROVIDER")
    llm_fast_model: str = Field(default="", alias="LLM_FAST_MODEL")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")

    # ── LangSmith ────────────────────────────────────────────────────────────
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = "mediai-langraph"
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_session_ttl: int = 3600  # seconds

    # ── Semantic Cache ────────────────────────────────────────────────────────
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.90  # Cosine similarity threshold (0.0 to 1.0)
    semantic_cache_ttl: int = 86400  # seconds (24 hours)

    # ── Chroma ───────────────────────────────────────────────────────────────
    chroma_mode: Literal["http", "local"] = "http"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "./chroma_db"

    # ── RAG ──────────────────────────────────────────────────────────────────
    knowledge_dir: str = "./knowledge"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    hf_token: str = Field(default="", alias="HF_TOKEN")


    # ── PubMed ───────────────────────────────────────────────────────────────
    pubmed_email: str = Field(default="", alias="PUBMED_EMAIL") # NCBI asks users to provide one (your email id) for API identification.
    pubmed_max_results: int = 10    # Maximum papers retrieved

    # ── Reflection ───────────────────────────────────────────────────────────
    max_reflection_iterations: int = 2 # Maximum number of reflection iterations (agent loop)

    # ── Testing ──────────────────────────────────────────────────────────────
    integration_tests: bool = False

    # ── Chroma collection names
    @property
    def chroma_collections(self) -> dict[str, str]:
        """Map subdirectory name → Chroma collection name."""
        return {
            "clinical_guidelines": "clinical_guidelines",
            "consensus_reports": "consensus_reports",
            "future_documents": "future_documents",
            "research_articles": "research_articles",
        }

    @field_validator("max_reflection_iterations")
    @classmethod
    def validate_reflection_iterations(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("max_reflection_iterations must be between 1 and 5")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
