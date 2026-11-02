import asyncio
import json

import httpx
import pytest

from product_card.llm import LLMBadOutput, LLMError
from product_card.models import NO_DATA, ProductCard, ProductInput
from product_card.pipeline import generate_card

from .fakes import (
    ScriptedLLM,
    dm02_product,
    draft_payload,
    draft_without_manufacturer,
    invented_range_payload,
    make_settings,
)


def run(llm: ScriptedLLM, product: ProductInput | None = None, **settings: object) -> ProductCard:
    return asyncio.run(
        generate_card(product or dm02_product(), llm, make_settings(**settings))  # type: ignore[arg-type]
    )


def issue_codes(card: ProductCard) -> set[str]:
    return {issue.code for issue in card.issues}


def test_valid_first_answer_makes_one_call_and_is_ready() -> None:
    llm = ScriptedLLM([draft_payload()])

    card = run(llm)

    assert card.status == "ready"
    assert len(llm.calls) == 1
    assert card.meta.attempts == 1
    assert card.issues == []
    assert card.missing_fields == []
    assert card.seo_title == draft_payload()["seo_title"]
    assert card.search_queries == draft_payload()["search_queries"]


def test_first_call_uses_the_cheap_model_chain() -> None:
    llm = ScriptedLLM([draft_payload()])

    run(llm)

    assert llm.calls[0]["models"] == make_settings().models


def test_invented_value_then_clean_answer_makes_two_calls() -> None:
    llm = ScriptedLLM([invented_range_payload(), draft_payload()])

    card = run(llm)

    assert card.status == "ready"
    assert len(llm.calls) == 2
    assert card.meta.attempts == 2
    assert card.issues == []


def test_repair_request_contains_previous_answer_and_error_text() -> None:
    first = invented_range_payload()
    llm = ScriptedLLM([first, draft_payload()])

    run(llm)

    repair = llm.calls[1]["messages"]
    assert [m["role"] for m in repair] == ["system", "user", "assistant", "user"]  # type: ignore[index]
    assert repair[2]["content"] == json.dumps(first, ensure_ascii=False)  # type: ignore[index]
    assert "2,5" in repair[3]["content"]  # type: ignore[index]
    assert "seo_description" in repair[3]["content"]  # type: ignore[index]


def test_permanent_failure_returns_needs_review_after_exactly_max_attempts() -> None:
    llm = ScriptedLLM([invented_range_payload()])

    card = run(llm, llm_max_attempts=3)

    assert card.status == "needs_review"
    assert len(llm.calls) == 3
    assert card.meta.attempts == 3
    assert {"unsupported_number"} <= issue_codes(card)
    assert all(issue.severity == "error" for issue in card.issues)
    assert card.seo_description == str(invented_range_payload()["seo_description"]).strip()
    assert card.seo_title == draft_payload()["seo_title"]


@pytest.mark.parametrize("attempts", [1, 2, 3, 4])
def test_number_of_calls_equals_llm_max_attempts(attempts: int) -> None:
    llm = ScriptedLLM([invented_range_payload()])

    run(llm, llm_max_attempts=attempts)

    assert len(llm.calls) == attempts


def test_last_attempt_uses_escalation_models() -> None:
    settings = make_settings(llm_max_attempts=3)
    llm = ScriptedLLM([invented_range_payload()])

    run(llm, llm_max_attempts=3)

    assert [call["models"] for call in llm.calls] == [
        settings.models,
        settings.models,
        settings.escalation_models,
    ]


def test_single_attempt_never_escalates() -> None:
    llm = ScriptedLLM([invented_range_payload()])

    run(llm, llm_max_attempts=1)

    assert llm.calls[0]["models"] == make_settings().models


def test_best_draft_is_kept_when_a_later_attempt_is_worse() -> None:
    worse = invented_range_payload()
    worse["seo_title"] = "а" * 71
    llm = ScriptedLLM([invented_range_payload(), worse, worse])

    card = run(llm)

    assert card.seo_title == draft_payload()["seo_title"]


def test_invalid_json_is_repaired() -> None:
    llm = ScriptedLLM(["this is not json", draft_payload()])

    card = run(llm)

    assert card.status == "ready"
    assert len(llm.calls) == 2
    repair = llm.calls[1]["messages"]
    assert repair[2]["content"] == "this is not json"  # type: ignore[index]
    assert "JSON" in repair[3]["content"]  # type: ignore[index]


def test_wrong_schema_is_repaired() -> None:
    llm = ScriptedLLM([{"seo_title": 5}, draft_payload()])

    card = run(llm)

    assert card.status == "ready"
    assert len(llm.calls) == 2


def test_json_in_code_fence_is_accepted() -> None:
    fenced = "```json\n" + json.dumps(draft_payload(), ensure_ascii=False) + "\n```"
    llm = ScriptedLLM([fenced])

    card = run(llm)

    assert card.status == "ready"
    assert len(llm.calls) == 1


def test_unusable_answers_on_every_attempt_raise_bad_output() -> None:
    llm = ScriptedLLM(["not json"])

    with pytest.raises(LLMBadOutput):
        run(llm, llm_max_attempts=3)

    assert len(llm.calls) == 3


def test_bad_output_error_from_client_is_retried_with_the_same_messages() -> None:
    llm = ScriptedLLM([LLMBadOutput(), draft_payload()])

    card = run(llm)

    assert card.status == "ready"
    assert len(llm.calls) == 2
    assert llm.calls[0]["messages"] == llm.calls[1]["messages"]


def test_bad_output_error_on_every_attempt_is_raised() -> None:
    llm = ScriptedLLM([LLMBadOutput()])

    with pytest.raises(LLMBadOutput):
        run(llm)

    assert len(llm.calls) == 3


@pytest.mark.parametrize(
    "product",
    [
        dm02_product(name=""),
        dm02_product(specs={}, description=""),
        dm02_product(specs={"диапазон": "н/д"}, description=""),
    ],
)
def test_insufficient_data_makes_no_llm_calls(product: ProductInput) -> None:
    llm = ScriptedLLM([draft_payload()])

    card = run(llm, product)

    assert card.status == "insufficient_data"
    assert llm.calls == []
    assert card.missing_fields
    assert card.seo_title == ""
    assert card.characteristics == []


def test_insufficient_data_keeps_notes_about_ignored_fields() -> None:
    llm = ScriptedLLM([draft_payload()])

    card = asyncio.run(
        generate_card(dm02_product(name=""), llm, make_settings(), ignored_fields=["price", "sku"])
    )

    assert "price, sku" in {item.field for item in card.missing_fields}


def test_api_failure_on_first_attempt_raises() -> None:
    llm = ScriptedLLM([LLMError("llm_auth", "Ключ не принят.", 502)])

    with pytest.raises(LLMError) as caught:
        run(llm)

    assert caught.value.code == "llm_auth"
    assert len(llm.calls) == 1


def test_non_llm_transport_error_on_first_attempt_propagates() -> None:
    llm = ScriptedLLM([httpx.ConnectError("down")])

    with pytest.raises(httpx.ConnectError):
        run(llm)


def test_api_failure_during_repair_returns_the_existing_draft() -> None:
    llm = ScriptedLLM([invented_range_payload(), LLMError("llm_unavailable", "Недоступно.", 503)])

    card = run(llm)

    assert card.status == "needs_review"
    assert len(llm.calls) == 2
    assert card.seo_description == str(invented_range_payload()["seo_description"]).strip()
    assert "unsupported_number" in issue_codes(card)


def test_characteristics_hold_exact_input_values_even_when_draft_text_is_wrong() -> None:
    card = run(ScriptedLLM([invented_range_payload()]))

    assert card.status == "needs_review"
    by_name = {c.name: c for c in card.characteristics}
    assert by_name["диапазон измерений"].value == "0–1,6 МПа"
    assert by_name["класс точности"].value == "1,5"
    assert by_name["присоединение"].value == "G1/2"
    assert by_name["диаметр корпуса"].value == "100 мм"
    assert by_name["производитель"].value == "Манотомь"
    assert all(c.source == "specs" for c in card.characteristics)


def test_llm_cannot_override_specs_through_extracted_specs() -> None:
    payload = draft_payload(
        extracted_specs=[
            {
                "name": "диапазон измерений",
                "value": "0–25 МПа",
                "quote": "Прибор предназначен для измерения избыточного давления",
            }
        ]
    )

    card = run(ScriptedLLM([payload]))

    ranges = [c for c in card.characteristics if c.name == "диапазон измерений"]
    assert [(c.value, c.source) for c in ranges] == [("0–1,6 МПа", "specs")]


GAP_DESCRIPTION = (
    "Прибор предназначен для измерения избыточного давления. Присоединение G1/2 штуцер радиальный."
)


def gap_product() -> ProductInput:
    specs = {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "диаметр корпуса": "100 мм",
    }
    return dm02_product(specs=specs, description=GAP_DESCRIPTION)


def test_extracted_spec_with_valid_quote_fills_the_gap() -> None:
    payload = draft_payload(
        extracted_specs=[
            {"name": "присоединение", "value": "G1/2", "quote": "Присоединение G1/2 штуцер"}
        ]
    )

    card = run(ScriptedLLM([payload]), gap_product())

    extracted = [c for c in card.characteristics if c.name == "присоединение"]
    assert [(c.value, c.source, c.quote) for c in extracted] == [
        ("G1/2", "description", "Присоединение G1/2 штуцер")
    ]
    assert "присоединение" not in {m.field for m in card.missing_fields}
    assert card.status == "ready"


def test_extracted_spec_with_fabricated_quote_is_dropped_with_a_warning() -> None:
    payload = draft_payload(
        extracted_specs=[
            {"name": "присоединение", "value": "G1/2", "quote": "Присоединение G1/2 с гайкой"}
        ]
    )

    card = run(ScriptedLLM([payload]), gap_product())

    gap = [c for c in card.characteristics if c.name == "присоединение"]
    assert [(c.value, c.source) for c in gap] == [(NO_DATA, "none")]
    assert [(i.severity, i.code) for i in card.issues] == [("warning", "extracted_spec_rejected")]
    assert "присоединение" in {m.field for m in card.missing_fields}
    assert card.status == "needs_clarification"


def test_clarification_from_the_model_gives_needs_clarification() -> None:
    payload = draft_payload(
        clarifications=[{"field": "диапазон", "question": "Какой диапазон верный?"}]
    )

    card = run(ScriptedLLM([payload]))

    assert card.status == "needs_clarification"
    assert [c.field for c in card.clarifications] == ["диапазон"]


def test_clarifications_are_limited_to_ten() -> None:
    payload = draft_payload(
        clarifications=[{"field": f"поле {i}", "question": "Что?"} for i in range(15)]
    )

    card = run(ScriptedLLM([payload]))

    assert len(card.clarifications) == 10


def test_missing_data_gives_needs_clarification() -> None:
    card = run(ScriptedLLM([draft_without_manufacturer()]), dm02_product(manufacturer=""))

    assert card.status == "needs_clarification"
    assert "manufacturer" in {m.field for m in card.missing_fields}


def test_manufacturer_absent_from_input_is_rejected_in_text() -> None:
    card = run(ScriptedLLM([draft_payload()]), dm02_product(manufacturer=""))

    assert card.status == "needs_review"
    assert any(issue.token == "Манотомь" for issue in card.issues)


def test_ignored_fields_are_reported_as_a_note() -> None:
    card = asyncio.run(
        generate_card(
            dm02_product(),
            ScriptedLLM([draft_payload()]),
            make_settings(),
            ignored_fields=["price"],
        )
    )

    assert card.status == "needs_clarification"
    assert "price" in {m.field for m in card.missing_fields}


def test_usage_and_cost_are_summed_over_attempts() -> None:
    llm = ScriptedLLM([invented_range_payload(), draft_payload()])

    card = run(llm)

    assert card.meta.prompt_tokens == 200
    assert card.meta.completion_tokens == 400
    assert card.meta.cost_usd == pytest.approx(0.0002)
    assert len(card.meta.models_used) == 2


def test_card_meta_carries_the_prompt_version() -> None:
    from product_card.prompts import PROMPT_VERSION

    card = run(ScriptedLLM([draft_payload()]))

    assert card.meta.prompt_version == PROMPT_VERSION
    assert card.meta.cached is False


def test_text_fields_are_trimmed() -> None:
    payload = draft_payload(seo_title="  Манометр DM 02  ")

    card = run(ScriptedLLM([payload]))

    assert card.seo_title == "Манометр DM 02"


def test_conflict_is_reported_even_if_the_model_misses_it():
    product = ProductInput(
        name="Манометр показывающий DM 02",
        manufacturer="Манотомь",
        specs={
            "диапазон измерений": "0–1,6 МПа",
            "класс точности": "1,5",
            "присоединение": "G1/2",
            "диаметр корпуса": "100 мм",
        },
        description="Прибор измеряет давление. По паспорту диапазон измерений 0–16 МПа.",
    )
    card = asyncio.run(generate_card(product, ScriptedLLM([draft_payload()]), make_settings()))
    assert card.status == "needs_clarification"
    assert any(item.field == "диапазон измерений" for item in card.clarifications)


def test_deadline_keeps_the_existing_draft():
    class SlowOnRepair(ScriptedLLM):
        async def complete(self, messages, *, models=None):
            if self.calls:
                await asyncio.sleep(1)
            return await super().complete(messages, models=models)

    llm = SlowOnRepair([invented_range_payload()])
    settings = make_settings(generation_deadline_seconds=0.05)
    card = asyncio.run(generate_card(dm02_product(), llm, settings))
    assert card.status == "needs_review"
    assert card.seo_title


def test_model_questions_about_brand_spelling_are_dropped():
    questions = [
        {"field": "Производитель", "question": "Верно ли написание бренда?"},
        {"field": "класс точности", "question": "Это класс или допуск?"},
    ]
    card = run(ScriptedLLM([draft_payload(clarifications=questions)]), dm02_product())
    assert [item.field for item in card.clarifications] == ["класс точности"]
