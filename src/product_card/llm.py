"""OpenRouter client with model routing, retries and error mapping."""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import Settings
from .prompts import DRAFT_JSON_SCHEMA, with_format_instruction

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_RETRY_DELAY_SECONDS = 20.0
# Free models limit requests per minute, so a short pause only wastes the retry.
_RATE_LIMIT_DELAY_SECONDS = 8.0


class LLMError(Exception):
    """An LLM call failed. `message` is safe to show: it holds no upstream body or secrets."""

    def __init__(self, code: str, message: str, http_status: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class LLMBadOutput(LLMError):
    """The API call worked, but its content is unusable. A new attempt may fix it."""

    def __init__(self, message: str = "Модель вернула ответ в неверном формате.") -> None:
        super().__init__("llm_bad_output", message, 502)


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None


class LLMClient(Protocol):
    async def complete(
        self, messages: list[dict[str, str]], *, models: list[str] | None = None
    ) -> LLMResult: ...

    async def aclose(self) -> None: ...


_STATUS_ERRORS: dict[int, tuple[str, str, int]] = {
    400: ("llm_bad_request", "Сервис LLM отклонил запрос. Проверьте настройки моделей.", 502),
    401: ("llm_auth", "Ключ OpenRouter не принят. Проверьте OPENROUTER_API_KEY в .env.", 502),
    402: ("llm_credits", "На балансе OpenRouter не хватает средств.", 502),
    403: ("llm_moderated", "Запрос отклонён модерацией или настройками доступа.", 502),
    404: (
        "llm_no_route",
        "Нет подходящей модели. Проверьте LLM_MODELS и настройки приватности.",
        502,
    ),
    408: ("llm_timeout", "Сервис LLM не ответил вовремя. Повторите запрос.", 504),
    429: ("llm_rate_limited", "Превышен лимит запросов к LLM. Повторите позже.", 429),
}
_DEFAULT_ERROR = ("llm_unavailable", "Сервис LLM временно недоступен. Повторите запрос.", 503)


def _error_for_status(status: int) -> LLMError:
    code, message, http_status = _STATUS_ERRORS.get(status, _DEFAULT_ERROR)
    return LLMError(code, message, http_status)


class OpenRouterClient:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        api_key = settings.openrouter_api_key
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value() if api_key else ''}",
                "X-Title": "Product Card AI",
            },
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            # The key must not follow a redirect to another host.
            follow_redirects=False,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_body(self, messages: list[dict[str, str]], models: list[str]) -> dict[str, Any]:
        settings = self._settings
        provider: dict[str, Any] = {
            "data_collection": settings.llm_data_collection,
            "max_price": {
                "prompt": settings.llm_max_price_prompt,
                "completion": settings.llm_max_price_completion,
            },
        }
        if settings.llm_zdr:
            provider["zdr"] = True
        body: dict[str, Any] = {
            # OpenRouter tries these in order on downtime, rate limits and moderation.
            "models": models,
            "messages": messages,
            "max_tokens": settings.llm_max_tokens,
            "reasoning": {"effort": settings.llm_reasoning_effort, "exclude": True},
            "provider": provider,
        }
        if "openrouter.ai" not in settings.openrouter_base_url:
            # A local OpenAI-compatible server such as Ollama takes one model and no routing.
            body = {
                "model": models[0],
                "messages": body["messages"],
                "max_tokens": settings.llm_max_tokens,
                "reasoning_effort": settings.llm_reasoning_effort,
            }
        if settings.llm_output_mode == "prompt":
            body["messages"] = with_format_instruction(messages)
            return body

        # Route only to endpoints that honor the JSON schema.
        if "provider" in body:
            provider["require_parameters"] = True
            body["plugins"] = [{"id": "response-healing"}]
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "product_card", "strict": True, "schema": DRAFT_JSON_SCHEMA},
        }
        return body

    async def complete(
        self, messages: list[dict[str, str]], *, models: list[str] | None = None
    ) -> LLMResult:
        body = self._build_body(messages, models or self._settings.models)
        last_error: LLMError | None = None
        retry_after: float | None = None

        for attempt in range(self._settings.llm_http_retries + 1):
            if attempt:
                rate_limited = last_error is not None and last_error.http_status == 429
                await asyncio.sleep(self._retry_delay(attempt, retry_after, rate_limited))
                retry_after = None
            try:
                response = await self._client.post("/chat/completions", json=body)
            except httpx.TimeoutException:
                last_error = LLMError("llm_timeout", "Сервис LLM не ответил вовремя.", 504)
                continue
            except httpx.HTTPError as exc:
                # The exception text may hold request details, so log only its type.
                logger.warning("LLM transport error: %s", type(exc).__name__)
                last_error = LLMError("llm_unavailable", "Нет связи с сервисом LLM.", 503)
                continue

            if response.status_code == 200:
                # OpenRouter may report an upstream failure inside a 200 response.
                payload = self._decode(response)
                error_status = self._embedded_error_status(payload)
                if error_status is None:
                    return self._parse_result(payload)
                last_error = _error_for_status(error_status)
                if error_status not in _RETRYABLE_STATUS:
                    raise last_error
                continue

            last_error = _error_for_status(response.status_code)
            logger.warning("LLM HTTP error: status=%s", response.status_code)
            if response.status_code not in _RETRYABLE_STATUS:
                raise last_error
            retry_after = self._parse_retry_after(response)

        raise last_error or LLMError(*_DEFAULT_ERROR)

    @staticmethod
    def _retry_delay(attempt: int, retry_after: float | None, rate_limited: bool = False) -> float:
        if retry_after is not None:
            return min(retry_after, _MAX_RETRY_DELAY_SECONDS)
        if rate_limited:
            return min(_RATE_LIMIT_DELAY_SECONDS * attempt, _MAX_RETRY_DELAY_SECONDS)
        backoff = min(2 ** (attempt - 1), _MAX_RETRY_DELAY_SECONDS)
        return float(backoff + random.uniform(0, 0.25))  # noqa: S311

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        try:
            return max(0.0, float(response.headers.get("retry-after", "")))
        except ValueError:
            return None

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMBadOutput("Сервис LLM вернул некорректный ответ.") from exc
        if not isinstance(payload, dict):
            raise LLMBadOutput("Сервис LLM вернул некорректный ответ.")
        return payload

    @staticmethod
    def _embedded_error_status(payload: dict[str, Any]) -> int | None:
        error = payload.get("error")
        choices = payload.get("choices")
        if not error and isinstance(choices, list) and choices and isinstance(choices[0], dict):
            error = choices[0].get("error")
        if not error:
            return None
        code = error.get("code") if isinstance(error, dict) else None
        return code if isinstance(code, int) else 502

    @staticmethod
    def _parse_result(payload: dict[str, Any]) -> LLMResult:
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMBadOutput("Сервис LLM вернул ответ без текста.") from exc
        if choice.get("finish_reason") == "length":
            raise LLMBadOutput("Ответ модели обрезан по лимиту токенов.")
        if not isinstance(content, str) or not content.strip():
            raise LLMBadOutput("Модель вернула пустой ответ.")

        raw_usage = payload.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        cost = usage.get("cost")
        return LLMResult(
            content=content,
            model=str(payload.get("model", "unknown")),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
        )
