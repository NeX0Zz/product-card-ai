import asyncio
from types import SimpleNamespace

import pytest

import product_card.service as service_module
from product_card.llm import LLMResult
from product_card.models import ProductCard, ProductInput
from product_card.service import CardService

from .fakes import (
    ScriptedLLM,
    dm02_product,
    draft_payload,
    draft_without_manufacturer,
    invented_range_payload,
    make_settings,
)


class TrackingLLM(ScriptedLLM):
    def __init__(self) -> None:
        super().__init__([draft_payload()])
        self.running = 0
        self.peak = 0

    async def complete(
        self, messages: list[dict[str, str]], *, models: list[str] | None = None
    ) -> LLMResult:
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(0.01)
            return await super().complete(messages, models=models)
        finally:
            self.running -= 1


def variant(label: str) -> ProductInput:
    return dm02_product(description=f"Прибор предназначен для измерения давления. Вариант {label}.")


def make_service(llm: ScriptedLLM, **settings: object) -> CardService:
    return CardService(llm, make_settings(**settings))  # type: ignore[arg-type]


def generate(service: CardService, product: ProductInput, **kwargs: bool) -> ProductCard:
    return asyncio.run(service.generate(product, **kwargs))


def test_second_identical_request_is_served_from_cache() -> None:
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> tuple[ProductCard, ProductCard]:
        service = make_service(llm)
        return await service.generate(dm02_product()), await service.generate(dm02_product())

    first, second = asyncio.run(scenario())

    assert len(llm.calls) == 1
    assert first.meta.cached is False
    assert second.meta.cached is True
    assert second.model_dump(exclude={"meta"}) == first.model_dump(exclude={"meta"})


def test_fresh_request_bypasses_the_cache() -> None:
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> ProductCard:
        service = make_service(llm)
        await service.generate(dm02_product())
        return await service.generate(dm02_product(), fresh=True)

    card = asyncio.run(scenario())

    assert len(llm.calls) == 2
    assert card.meta.cached is False


def test_fresh_result_replaces_the_cached_one() -> None:
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> ProductCard:
        service = make_service(llm)
        await service.generate(dm02_product())
        await service.generate(dm02_product(), fresh=True)
        return await service.generate(dm02_product())

    assert asyncio.run(scenario()).meta.cached is True
    assert len(llm.calls) == 2


def test_different_products_do_not_share_cache_entries() -> None:
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> None:
        service = make_service(llm)
        await service.generate(variant("альфа"))
        await service.generate(variant("бета"))

    asyncio.run(scenario())

    assert len(llm.calls) == 2


def test_cache_entry_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(service_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> list[bool]:
        service = make_service(llm, cache_ttl_seconds=60)
        results = [await service.generate(dm02_product())]
        now[0] += 59
        results.append(await service.generate(dm02_product()))
        now[0] += 2
        results.append(await service.generate(dm02_product()))
        return [card.meta.cached for card in results]

    assert asyncio.run(scenario()) == [False, True, False]
    assert len(llm.calls) == 2


def test_least_recently_used_entry_is_evicted() -> None:
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> dict[str, bool]:
        service = make_service(llm, cache_max_entries=2)
        await service.generate(variant("альфа"))
        await service.generate(variant("бета"))
        await service.generate(variant("альфа"))
        await service.generate(variant("гамма"))
        calls_before = len(llm.calls)
        alpha = await service.generate(variant("альфа"))
        beta = await service.generate(variant("бета"))
        return {
            "calls_before": calls_before == 3,
            "alpha_cached": alpha.meta.cached,
            "beta_cached": beta.meta.cached,
        }

    assert asyncio.run(scenario()) == {
        "calls_before": True,
        "alpha_cached": True,
        "beta_cached": False,
    }


def test_needs_review_is_not_cached() -> None:
    llm = ScriptedLLM([invented_range_payload()])

    async def scenario() -> tuple[ProductCard, ProductCard]:
        service = make_service(llm, llm_max_attempts=2)
        return await service.generate(dm02_product()), await service.generate(dm02_product())

    first, second = asyncio.run(scenario())

    assert first.status == second.status == "needs_review"
    assert second.meta.cached is False
    assert len(llm.calls) == 4


def test_insufficient_data_is_not_cached() -> None:
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> ProductCard:
        service = make_service(llm)
        await service.generate(dm02_product(name=""))
        return await service.generate(dm02_product(name=""))

    card = asyncio.run(scenario())

    assert card.status == "insufficient_data"
    assert card.meta.cached is False


def test_needs_clarification_is_cached() -> None:
    llm = ScriptedLLM([draft_without_manufacturer()])

    async def scenario() -> ProductCard:
        service = make_service(llm)
        await service.generate(dm02_product(manufacturer=""))
        return await service.generate(dm02_product(manufacturer=""))

    card = asyncio.run(scenario())

    assert card.status == "needs_clarification"
    assert card.meta.cached is True
    assert len(llm.calls) == 1


def test_concurrent_identical_requests_make_one_llm_call() -> None:
    llm = TrackingLLM()

    async def scenario() -> list[ProductCard]:
        service = make_service(llm)
        return list(await asyncio.gather(*(service.generate(dm02_product()) for _ in range(6))))

    cards = asyncio.run(scenario())

    assert len(llm.calls) == 1
    assert {card.status for card in cards} == {"ready"}


def test_in_flight_request_is_released_after_completion() -> None:
    llm = ScriptedLLM([draft_payload()])

    async def scenario() -> int:
        service = make_service(llm)
        await service.generate(dm02_product())
        await asyncio.sleep(0)
        return len(service._in_flight)

    assert asyncio.run(scenario()) == 0


def test_cancelled_waiter_does_not_cancel_the_shared_generation() -> None:
    llm = TrackingLLM()

    async def scenario() -> ProductCard:
        service = make_service(llm)
        first = asyncio.create_task(service.generate(dm02_product()))
        second = asyncio.create_task(service.generate(dm02_product()))
        await asyncio.sleep(0)
        first.cancel()
        return await second

    assert asyncio.run(scenario()).status == "ready"
    assert len(llm.calls) == 1


def test_failure_is_propagated_to_every_waiter_and_not_cached() -> None:
    from product_card.llm import LLMError

    llm = ScriptedLLM([LLMError("llm_credits", "Нет средств.", 502)])

    async def scenario() -> list[object]:
        service = make_service(llm)
        results = await asyncio.gather(
            service.generate(dm02_product()),
            service.generate(dm02_product()),
            return_exceptions=True,
        )
        retry = await asyncio.gather(service.generate(dm02_product()), return_exceptions=True)
        return [*results, *retry]

    results = asyncio.run(scenario())

    assert all(isinstance(item, LLMError) for item in results)
    assert len(llm.calls) == 2


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_concurrency_limit_holds(limit: int) -> None:
    llm = TrackingLLM()

    async def scenario() -> None:
        service = make_service(llm, max_concurrent_generations=limit)
        labels = ["альфа", "бета", "гамма", "дельта", "эпсилон", "дзета", "эта", "тета"]
        await asyncio.gather(*(service.generate(variant(label)) for label in labels))

    asyncio.run(scenario())

    assert len(llm.calls) == 8
    assert llm.peak == limit


def test_cache_key_depends_on_the_models() -> None:
    product = dm02_product()
    llm = ScriptedLLM([draft_payload()])

    default = make_service(llm)._key(product)
    other_models = make_service(llm, llm_models="vendor/other")._key(product)
    other_escalation = make_service(llm, llm_escalation_models="vendor/strong")._key(product)

    assert len({default, other_models, other_escalation}) == 3


def test_cache_key_is_stable_for_equal_products() -> None:
    llm = ScriptedLLM([draft_payload()])
    service = make_service(llm)

    assert service._key(dm02_product()) == service._key(dm02_product())


def test_cached_copy_reports_no_spending():
    llm = ScriptedLLM([draft_payload()])
    service = CardService(llm, make_settings())

    async def run() -> tuple[ProductCard, ProductCard]:
        return await service.generate(dm02_product()), await service.generate(dm02_product())

    first, second = asyncio.run(run())
    assert first.meta.cost_usd is not None
    assert second.meta.cached is True
    assert second.meta.cost_usd is None


def test_ignored_fields_are_part_of_the_cache_key():
    llm = ScriptedLLM([draft_payload()])
    service = CardService(llm, make_settings())

    async def run() -> None:
        await service.generate(dm02_product(), ["sku"])
        await service.generate(dm02_product(), ["price"])

    asyncio.run(run())
    assert len(llm.calls) == 2
