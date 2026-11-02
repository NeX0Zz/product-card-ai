"""Local web service: JSON API, static UI and security controls."""

import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .config import Settings, get_settings
from .llm import LLMClient, LLMError, OpenRouterClient
from .models import InputError, ProductCard, load_json, parse_product
from .pipeline import precheck_card
from .render import to_markdown, to_text
from .service import CardService

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "input"

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; manifest-src 'self'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}


def _apply_security_headers(response: Response, path: str) -> None:
    response.headers.update(_SECURITY_HEADERS)
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"


def _error(status: int, code: str, message: str, details: list[Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details or []}},
    )


def _card_response(card: ProductCard) -> dict[str, Any]:
    return {
        "card": card.model_dump(),
        "exports": {"markdown": to_markdown(card), "text": to_text(card)},
    }


class RateLimiter:
    """Sliding window per client. Every generation request costs real money."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def retry_after(self, client: str) -> int:
        """Register a request. Returns 0 if allowed, otherwise seconds to wait."""
        now = time.monotonic()
        hits = self._hits.setdefault(client, deque())
        while hits and now - hits[0] > self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return int(self._window - (now - hits[0])) + 1
        hits.append(now)
        return 0


def _origin_allowed(request: Request) -> bool:
    """Reject cross-site browser requests, which could spend the key from another page.

    Another port on localhost is another origin too, so the port is compared as well.
    """
    if request.headers.get("sec-fetch-site", "same-origin") not in {"same-origin", "none"}:
        return False
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).netloc == request.headers.get("host")


async def _read_limited_body(request: Request, limit: int) -> bytes | None:
    """Read the body with a hard limit, so a wrong Content-Length cannot bypass it."""
    declared = request.headers.get("content-length", "0")
    if declared.isdigit() and int(declared) > limit:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _load_examples() -> list[dict[str, Any]]:
    examples = []
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        try:
            examples.append({"id": path.stem, "data": load_json(path.read_text("utf-8"))})
        except (OSError, InputError):
            logger.warning("Skipped broken example: %s", path.name)
    return examples


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    configured = llm is not None or settings.openrouter_api_key is not None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = llm
        if client is None and settings.openrouter_api_key is not None:
            client = OpenRouterClient(settings)
        app.state.service = CardService(client, settings) if client else None
        yield
        if client is not None:
            await client.aclose()

    # The interactive docs need third-party scripts, which the CSP forbids.
    app = FastAPI(
        title="Product Card AI",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.hosts)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # Registered last, so it wraps the others: a rejected Host still gets these headers.
    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        _apply_security_headers(response, request.url.path)
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        # The default handler echoes user input in its response.
        return _error(422, "invalid_request", "Некорректные параметры запроса.")

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", "Запрос не может быть выполнен.")

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unexpected error: %s", type(exc).__name__)
        response = _error(500, "internal_error", "Внутренняя ошибка сервиса. Подробности в логе.")
        # This handler runs outside all middleware, so the headers are set here.
        _apply_security_headers(response, request.url.path)
        return response

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "llm_configured": configured,
            "models": settings.models,
            "escalation_models": settings.escalation_models,
        }

    @app.get("/api/examples")
    async def examples() -> dict[str, Any]:
        return {"examples": _load_examples()}

    @app.post("/api/generate", response_model=None)
    async def generate(request: Request, fresh: bool = False) -> JSONResponse | dict[str, Any]:
        if not _origin_allowed(request):
            return _error(403, "forbidden_origin", "Запрос с другого сайта отклонён.")

        # JSON only: browsers can't send it cross-site without a CORS preflight.
        content_type = request.headers.get("content-type", "").split(";")[0].strip().casefold()
        if content_type != "application/json":
            return _error(415, "unsupported_media_type", "Нужен Content-Type: application/json.")

        client = request.client.host if request.client else "unknown"
        wait = limiter.retry_after(client)
        if wait:
            response = _error(
                429, "rate_limited", f"Слишком много запросов. Повторите через {wait} с."
            )
            response.headers["Retry-After"] = str(wait)
            return response

        body = await _read_limited_body(request, settings.max_body_bytes)
        if body is None:
            limit_kb = settings.max_body_bytes // 1000
            return _error(413, "body_too_large", f"Запрос больше {limit_kb} КБ.")

        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return _error(400, "invalid_json", "Запрос должен быть в кодировке UTF-8.")

        try:
            product, ignored = parse_product(load_json(text))
        except InputError as exc:
            status = 400 if exc.code == "invalid_json" else 422
            details = [item.model_dump() for item in exc.details]
            return _error(status, exc.code, exc.message, details)

        insufficient = precheck_card(product, ignored)
        if insufficient is not None:
            return _card_response(insufficient)

        service: CardService | None = request.app.state.service
        if service is None:
            return _error(
                503,
                "llm_not_configured",
                "Не задан OPENROUTER_API_KEY. Добавьте ключ в файл .env и перезапустите сервис.",
            )

        try:
            card = await service.generate(product, ignored, fresh=fresh)
        except LLMError as exc:
            logger.warning("Generation failed: %s", exc.code)
            return _error(exc.http_status, exc.code, exc.message)
        except httpx.HTTPError as exc:
            logger.warning("Generation failed: %s", type(exc).__name__)
            return _error(503, "llm_unavailable", "Нет связи с сервисом LLM.")

        logger.info(
            "generated status=%s attempts=%s cached=%s duration_ms=%s",
            card.status,
            card.meta.attempts,
            card.meta.cached,
            card.meta.duration_ms,
        )
        return _card_response(card)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
