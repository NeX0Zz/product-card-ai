"""Spend control around the pipeline: result cache, request merging, concurrency limit."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from functools import partial

from .config import Settings
from .llm import LLMClient
from .models import ProductCard, ProductInput
from .pipeline import generate_card
from .prompts import PROMPT_VERSION


class CardService:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_generations)
        self._cache: OrderedDict[str, tuple[float, ProductCard]] = OrderedDict()
        self._in_flight: dict[str, asyncio.Task[ProductCard]] = {}

    def _key(self, product: ProductInput, ignored: list[str] | None = None) -> str:
        payload = {
            "product": product.model_dump(),
            "ignored": ignored or [],
            "prompt": PROMPT_VERSION,
            "models": [self._settings.models, self._settings.escalation_models],
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _cache_get(self, key: str) -> ProductCard | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        created, card = entry
        if time.monotonic() - created > self._settings.cache_ttl_seconds:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        # This request spent nothing, so the copy doesn't report the original cost.
        meta = card.meta.model_copy(
            update={"cached": True, "cost_usd": None, "prompt_tokens": 0, "completion_tokens": 0}
        )
        return card.model_copy(update={"meta": meta})

    def _cache_put(self, key: str, card: ProductCard) -> None:
        # Failed drafts stay out, so a new request gets a new chance.
        if card.status not in {"ready", "needs_clarification"}:
            return
        self._cache[key] = (time.monotonic(), card)
        self._cache.move_to_end(key)
        while len(self._cache) > self._settings.cache_max_entries:
            self._cache.popitem(last=False)

    async def _run(self, key: str, product: ProductInput, ignored: list[str]) -> ProductCard:
        async with self._semaphore:
            card = await generate_card(product, self._llm, self._settings, ignored)
        self._cache_put(key, card)
        return card

    def _forget_task(self, key: str, task: asyncio.Task[ProductCard]) -> None:
        self._in_flight.pop(key, None)
        # If every waiter disconnected, nobody reads this exception, so mark it as handled.
        if not task.cancelled():
            task.exception()

    async def generate(
        self, product: ProductInput, ignored: list[str] | None = None, *, fresh: bool = False
    ) -> ProductCard:
        ignored = ignored or []
        key = self._key(product, ignored)
        if not fresh:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        # A double click or a parallel identical request must not pay twice.
        task = self._in_flight.get(key)
        if task is None:
            task = asyncio.create_task(self._run(key, product, ignored))
            self._in_flight[key] = task
            task.add_done_callback(partial(self._forget_task, key))
        return await asyncio.shield(task)
