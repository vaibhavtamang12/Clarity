"""LLM provider factory — provider choice is configuration (Rule 3)."""

from __future__ import annotations

from app.core.config import LLMSettings
from app.core.exceptions import ConfigurationError
from app.llm.base import LLMProvider
from app.llm.openai_compatible import OpenAICompatibleProvider

_DEFAULT_BASE_URLS = {
    "openai_compatible": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
}


def build_llm_provider(settings: LLMSettings) -> LLMProvider:
    if settings.provider in ("openai_compatible", "ollama"):
        base_url = settings.base_url or _DEFAULT_BASE_URLS[settings.provider]
        return OpenAICompatibleProvider(
            model=settings.model,
            base_url=base_url,
            api_key=settings.api_key.get_secret_value() if settings.api_key else None,
            timeout_seconds=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
    if settings.provider == "anthropic":
        raise ConfigurationError(
            "Anthropic-native adapter lands with the generation pipeline (Phase 11); "
            "use an OpenAI-compatible gateway until then"
        )
    raise ConfigurationError(f"Unknown LLM provider: {settings.provider}")