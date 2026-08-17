"""OpenAI-compatible chat-completions adapter.

One transport covers OpenAI, Ollama (its /v1 endpoint), vLLM, LiteLLM, and
any compatible gateway — provider choice is a base_url + model config change
(Rule 3). Anthropic-native support lands with the generation pipeline
(Phase 11, D-055).

Resilience:
- Retries with backoff on timeouts and retryable statuses (429/5xx).
- Non-retryable 4xx fails fast with a typed error.
- Token usage parsed from the response — cost accounting is mandatory,
  not optional (feeds Phase 23).
"""

from __future__ import annotations

import asyncio
import time

import httpx

from app.core.exceptions import LLMUnavailableError
from app.core.logging import get_logger
from app.llm.base import LLMRequest, LLMResponse, TokenUsage

logger = get_logger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
        client: httpx.AsyncClient | None = None,  # injectable for tests
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_base_delay_seconds
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------- public
    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload: dict = {
            "model": self._model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_output:
            payload["response_format"] = {"type": "json_object"}
        if request.stop:
            payload["stop"] = request.stop

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempts = self._max_retries + 1
        last_error: Exception | None = None
        start = time.perf_counter()

        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.post(self._url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = exc
                logger.warning(
                    "llm_request_transport_error",
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                if attempt < attempts:
                    await asyncio.sleep(self._retry_delay * (2 ** (attempt - 1)))
                continue

            if response.status_code in _RETRYABLE_STATUS:
                last_error = LLMUnavailableError(
                    f"LLM returned retryable status {response.status_code}"
                )
                logger.warning("llm_request_retryable_status", status=response.status_code, attempt=attempt)
                if attempt < attempts:
                    await asyncio.sleep(self._retry_delay * (2 ** (attempt - 1)))
                continue

            if response.status_code >= 400:
                raise LLMUnavailableError(
                    f"LLM request failed with status {response.status_code}"
                )

            latency_ms = (time.perf_counter() - start) * 1000
            return self._parse(response, latency_ms)

        raise LLMUnavailableError(
            f"LLM request failed after {attempts} attempts"
        ) from last_error

    # ---------------------------------------------------------------- internals
    def _parse(self, response: httpx.Response, latency_ms: float) -> LLMResponse:
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMUnavailableError("LLM returned a malformed response") from exc
        usage_raw = data.get("usage") or {}
        return LLMResponse(
            content=str(content),
            model=data.get("model") or self._model,
            usage=TokenUsage(
                prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
                completion_tokens=int(usage_raw.get("completion_tokens", 0)),
            ),
            latency_ms=round(latency_ms, 2),
        )