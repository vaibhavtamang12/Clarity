"""OpenAI-compatible chat-completions adapter with streaming support."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from app.core.exceptions import LLMUnavailableError
from app.core.logging import get_logger
from app.llm.base import LLMRequest, LLMResponse, StreamEvent, TokenUsage

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
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_base_delay_seconds
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Non-streaming generation (existing implementation)."""
        payload: dict = {
            "model": self._model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
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
                logger.warning("llm_request_transport_error", attempt=attempt, error=type(exc).__name__)
                if attempt < attempts:
                    await asyncio.sleep(self._retry_delay * (2 ** (attempt - 1)))
                continue

            if response.status_code in _RETRYABLE_STATUS:
                last_error = LLMUnavailableError(f"LLM returned retryable status {response.status_code}")
                logger.warning("llm_request_retryable_status", status=response.status_code, attempt=attempt)
                if attempt < attempts:
                    await asyncio.sleep(self._retry_delay * (2 ** (attempt - 1)))
                continue

            if response.status_code >= 400:
                raise LLMUnavailableError(f"LLM request failed with status {response.status_code}")

            latency_ms = (time.perf_counter() - start) * 1000
            return self._parse(response, latency_ms)

        raise LLMUnavailableError(f"LLM request failed after {attempts} attempts") from last_error

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Streaming generation — yields tokens incrementally."""
        payload: dict = {
            "model": self._model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
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

        for attempt in range(1, attempts + 1):
            try:
                async with self._client.stream("POST", self._url, json=payload, headers=headers) as response:
                    if response.status_code in _RETRYABLE_STATUS:
                        last_error = LLMUnavailableError(f"LLM returned retryable status {response.status_code}")
                        logger.warning("llm_stream_retryable_status", status=response.status_code, attempt=attempt)
                        if attempt < attempts:
                            await asyncio.sleep(self._retry_delay * (2 ** (attempt - 1)))
                        continue

                    if response.status_code >= 400:
                        await response.aread()
                        raise LLMUnavailableError(f"LLM stream failed with status {response.status_code}")

                    # Parse SSE stream
                    async for event in self._parse_stream(response):
                        yield event
                    return  # success

            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = exc
                logger.warning("llm_stream_transport_error", attempt=attempt, error=type(exc).__name__)
                if attempt < attempts:
                    await asyncio.sleep(self._retry_delay * (2 ** (attempt - 1)))
                continue

        yield StreamEvent(error=f"LLM stream failed after {attempts} attempts: {type(last_error).__name__}", done=True)

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

    async def _parse_stream(self, response: httpx.Response) -> AsyncIterator[StreamEvent]:
        """Parse OpenAI SSE stream format."""
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                yield StreamEvent(done=True)
                return
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield StreamEvent(token=content, model=chunk.get("model"))
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                logger.warning("llm_stream_parse_error", error=type(exc).__name__)
                continue