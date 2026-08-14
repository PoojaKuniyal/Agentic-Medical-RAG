"""
LLM provider factory with automatic resilience.

This file is acting as an LLM Factory.

The idea is:
All agents call `get_llm()` — they have zero knowledge of the underlying
provider. Swapping LLM_PROVIDER in .env is sufficient to change the model
globally without touching any agent code.

Supported providers
  • google_genai  — Gemini via Google AI Studio
  • openai        — OpenAI Chat models
  • anthropic     — Anthropic Claude models
  • groq          — Groq Cloud models
  • ollama        — Local open-source models via Ollama
  • open_source   — Open-source models via OpenAI-compatible endpoints (vLLM, LM Studio, etc.)
"""

from functools import lru_cache
import logging
import time
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import get_settings

logger = logging.getLogger(__name__)


class ResilientChatModel(BaseChatModel):
    """
    Wrapper chat model that calls the primary LLM provider (Google, OpenAI, Anthropic, Groq, Ollama, OpenSource)
    and automatically retries with backoff if rate limits or transient errors occur.
    """

    primary_llm: BaseChatModel
    provider_name: str = "open_source"

    @property
    def _llm_type(self) -> str:
        return f"resilient_{self.provider_name}"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Delegate structured output binding directly to the primary LLM provider."""
        if "method" not in kwargs and self.provider_name in ("groq", "ollama", "open_source"):
            kwargs["method"] = "function_calling"
        return self.primary_llm.with_structured_output(schema, **kwargs)



    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        callbacks = (
            run_manager.get_child()
            if run_manager and hasattr(run_manager, "get_child")
            else None
        )
        config = {"callbacks": callbacks} if callbacks else None

        # Total API attempts, including the initial request
        max_attempts = 3 
        for attempt in range(1, max_attempts + 1):
            try:
                response_message = self.primary_llm.invoke(
                    messages, stop=stop, config=config, **kwargs
                )
                return ChatResult(
                    generations=[ChatGeneration(message=response_message)]
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                is_retryable = any(
                    err in exc_str
                    for err in [
                    # Rate limiting / temporary throttling
                    "429",
                    "rate_limit",
                    "resource_exhausted",

                    # Temporary server errors
                    "500",
                    "502",
                    "503",
                    "504",

                    # Network / timeout errors
                    "timeout",
                    "timed out",
                    "connection reset",
                                    ]
                )
                if is_retryable and attempt < max_attempts:
                    wait_seconds = attempt * 15 # backoff for free tier RPM rate limits
                    logger.warning(
                        "[LLM Factory] Rate limit / quota 429 encountered on attempt %d/%d for '%s'. Waiting %ds before retry...",
                        attempt,
                        max_attempts,
                        self.provider_name,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                logger.error(
                    "[LLM Factory] Provider error (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                raise exc


# Backward compatibility alias
SmartFallbackChatModel = ResilientChatModel


def get_llm(**kwargs: Any) -> BaseChatModel:
    """
    Return a configured LangChain chat model instance wrapped in ResilientChatModel.
    """
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "google_genai":
        primary_llm = _build_google_genai(settings, **kwargs)
    elif provider == "openai":
        primary_llm = _build_openai(settings, **kwargs)
    elif provider == "anthropic":
        primary_llm = _build_anthropic(settings, **kwargs)
    elif provider == "groq":
        primary_llm = _build_groq(settings, **kwargs)
    elif provider == "ollama":
        primary_llm = _build_ollama(settings, **kwargs)
    elif provider == "open_source":
        primary_llm = _build_open_source(settings, **kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider}'. "
            "Supported values: google_genai, openai, anthropic, groq, ollama, open_source"
        )

    return ResilientChatModel(
        primary_llm=primary_llm,
        provider_name=provider,
    )


def _build_google_genai(settings: Any, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(
            "langchain-google-genai is required for LLM_PROVIDER=google_genai. "
            "Install it with: pip install langchain-google-genai"
        ) from e

    defaults = dict(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
        max_retries=0,
    )
    defaults.update(kwargs)
    return ChatGoogleGenerativeAI(**defaults)


def _build_openai(settings: Any, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for LLM_PROVIDER=openai. "
            "Install it with: pip install langchain-openai"
        ) from e

    defaults = dict(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        max_retries=0,
    )
    defaults.update(kwargs)
    return ChatOpenAI(**defaults)


def _build_anthropic(settings: Any, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError(
            "langchain-anthropic is required for LLM_PROVIDER=anthropic. "
            "Install it with: pip install langchain-anthropic"
        ) from e

    defaults = dict(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0.1,
        max_retries=0,
    )
    defaults.update(kwargs)
    return ChatAnthropic(**defaults)


def _build_groq(settings: Any, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        # Fallback to langchain_openai with Groq base URL
        try:
            from langchain_openai import ChatOpenAI
            defaults = dict(
                model=settings.llm_model,
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.1,
                max_retries=0,
            )
            defaults.update(kwargs)
            return ChatOpenAI(**defaults)
        except ImportError as e:
            raise ImportError(
                "langchain-groq or langchain-openai is required for LLM_PROVIDER=groq."
            ) from e

    defaults = dict(
        model=settings.llm_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.1,
        max_retries=0,
    )
    defaults.update(kwargs)
    return ChatGroq(**defaults)


def _resolve_base_url_for_docker(url: str) -> str:
    """If running inside a Docker container, map localhost/127.0.0.1 to host.docker.internal."""
    import os
    if os.path.exists("/.dockerenv") and url:
        return url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
    return url


def _build_ollama(settings: Any, **kwargs: Any) -> BaseChatModel:
    base_url = _resolve_base_url_for_docker(settings.ollama_base_url)
    try:
        from langchain_community.chat_models import ChatOllama
        defaults = dict(
            model=settings.llm_model,
            base_url=base_url,
            temperature=0.1,
        )
        defaults.update(kwargs)
        return ChatOllama(**defaults)
    except ImportError:
        from langchain_openai import ChatOpenAI
        defaults = dict(
            model=settings.llm_model,
            api_key="ollama",
            base_url=f"{base_url.rstrip('/')}/v1",
            temperature=0.1,
        )
        defaults.update(kwargs)
        return ChatOpenAI(**defaults)


def _build_open_source(settings: Any, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for LLM_PROVIDER=open_source."
        ) from e

    base_url = _resolve_base_url_for_docker(settings.openai_base_url)
    defaults = dict(
        model=settings.llm_model,
        api_key=settings.openai_api_key or "open-source",
        base_url=base_url,
        temperature=0.1,
        max_retries=0,
    )
    defaults.update(kwargs)
    return ChatOpenAI(**defaults)


@lru_cache(maxsize=1)
def get_cached_llm() -> BaseChatModel:
    return get_llm()
