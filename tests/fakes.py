"""Scripted LLM for tests and for UI work without an API key. Not used in production."""

import json
from collections.abc import Iterable

from product_card.config import Settings
from product_card.llm import LLMResult
from product_card.models import ProductInput, parse_product


def draft_payload(**overrides: object) -> dict[str, object]:
    """A draft for DM 02 from examples/input that passes validation."""
    filler = (
        "Перед заказом сверьте диапазон измерений, класс точности и присоединение с условиями "
        "своей задачи: от этих параметров зависит, подойдёт ли прибор для конкретной точки "
        "измерения и совпадёт ли он с ответной частью трубопровода. "
    )
    payload: dict[str, object] = {
        "seo_title": "Манометр показывающий DM 02 Манотомь, 0–1,6 МПа",
        "seo_description": (
            "Манометр показывающий DM 02 производства Манотомь предназначен для измерения "
            "избыточного давления. Диапазон измерений составляет 0–1,6 МПа, класс точности 1,5, "
            "присоединение G1/2, диаметр корпуса 100 мм. " + filler * 4
        ),
        "search_queries": [
            "манометр показывающий dm 02",
            "манометр dm 02 манотомь",
            "манометр 0-1,6 мпа g1/2",
        ],
        "commercial": {
            "suitable_for": ["Служба КИП предприятия", "Инженер, подбирающий манометр на замену"],
            "solves": ["Контроль избыточного давления", "Замена прибора с теми же параметрами"],
        },
        "extracted_specs": [],
        "clarifications": [],
    }
    payload.update(overrides)
    return payload


def draft_without_manufacturer() -> dict[str, object]:
    """The same draft for a product whose input names no manufacturer."""
    text = json.dumps(draft_payload(), ensure_ascii=False)
    return json.loads(
        text.replace(" производства Манотомь", "").replace(" Манотомь", "").replace(" манотомь", "")
    )


class ScriptedLLM:
    """Returns prepared answers in order and records every call."""

    def __init__(self, answers: Iterable[object]) -> None:
        self._answers = list(answers)
        self.calls: list[dict[str, object]] = []

    async def complete(
        self, messages: list[dict[str, str]], *, models: list[str] | None = None
    ) -> LLMResult:
        self.calls.append({"messages": messages, "models": models})
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        if isinstance(answer, Exception):
            raise answer
        content = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return LLMResult(
            content=content,
            model=(models or ["fake/model"])[0],
            prompt_tokens=100,
            completion_tokens=200,
            cost_usd=0.0001,
        )

    async def aclose(self) -> None:
        return None


DM02_RAW: dict[str, object] = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": (
        "Прибор предназначен для измерения избыточного давления в технологических средах."
    ),
}


def dm02_raw(**overrides: object) -> dict[str, object]:
    return {**DM02_RAW, **overrides}


def dm02_product(**overrides: object) -> ProductInput:
    product, _ = parse_product(dm02_raw(**overrides))
    return product


def make_settings(**overrides: object) -> Settings:
    """Settings that ignore a developer's .env file."""
    # Fixed routing keeps tests independent from the default model chain.
    defaults: dict[str, object] = {
        "llm_output_mode": "schema",
        "llm_data_collection": "deny",
        "llm_models": "test/primary,test/fallback",
        "llm_escalation_models": "test/strong",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})  # type: ignore[arg-type]


def invented_range_payload() -> dict[str, object]:
    """A draft whose description states a range that the DM 02 input does not contain."""
    payload = draft_payload()
    description = str(payload["seo_description"]).replace("0–1,6 МПа", "0–2,5 МПа")
    payload["seo_description"] = description
    return payload
