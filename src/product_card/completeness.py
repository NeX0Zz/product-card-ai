"""Deterministic input check: which expected fields are absent, before any LLM call.

Expected fields depend on the instrument type: a mechanical pressure gauge has no power
supply, so asking for one would only produce noise for the content manager.
"""

import re
from dataclasses import dataclass, field

from .models import NO_DATA, Characteristic, MissingField, ProductInput

# Values people put in spreadsheets instead of leaving a cell empty.
PLACEHOLDERS = {
    "",
    "-",
    "--",
    "—",
    "–",
    "?",
    "н/д",
    "нд",
    "n/a",
    "na",
    "tbd",
    "null",
    "none",
    "нет данных",
    "данных нет",
    "уточняется",
    "уточнить",
    "не указано",
    "неизвестно",
    "unknown",
}

MIN_DESCRIPTION_CHARS = 40


@dataclass(frozen=True)
class ExpectedSpec:
    label: str
    aliases: tuple[str, ...]

    def matches(self, text: str) -> bool:
        lowered = text.casefold()
        return any(alias in lowered for alias in self.aliases)


RANGE = ExpectedSpec("диапазон измерений", ("диапазон", "предел", "шкал"))
ACCURACY = ExpectedSpec("класс точности", ("класс точности", "точност", "погрешност"))
POWER = ExpectedSpec("питание", ("питани", "напряжени"))
CASE = ExpectedSpec("тип корпуса", ("корпус",))
CONNECTION = ExpectedSpec("присоединение", ("присоединени", "резьб", "штуцер"))
INTERFACES = ExpectedSpec(
    "интерфейсы", ("интерфейс", "выходной сигнал", "выход", "протокол", "rs-485", "modbus", "hart")
)

_ELECTRONIC_HINT = re.compile(r"цифров|электро|датчик|преобразовател|контроллер|модул", re.I)
_MECHANICAL_NAME = re.compile(
    r"манометр|вакуумметр|мановакуумметр|напоромер|тягомер|термометр", re.I
)


def expected_specs(product: ProductInput) -> tuple[ExpectedSpec, ...]:
    if _MECHANICAL_NAME.search(product.name) and not _ELECTRONIC_HINT.search(product.name):
        return (RANGE, ACCURACY, CONNECTION, CASE)
    return (RANGE, ACCURACY, POWER, CASE, INTERFACES)


def is_placeholder(value: str) -> bool:
    return value.strip().casefold() in PLACEHOLDERS


@dataclass
class Completeness:
    """`gaps` holds expected fields without a value. The LLM may later fill them from the text."""

    blocking: bool = False
    missing: list[MissingField] = field(default_factory=list)
    gaps: list[ExpectedSpec] = field(default_factory=list)
    filled_specs: dict[str, str] = field(default_factory=dict)


def check_completeness(product: ProductInput) -> Completeness:
    result = Completeness()
    result.filled_specs = {k: v for k, v in product.specs.items() if not is_placeholder(v)}

    if not product.name:
        result.blocking = True
        result.missing.append(
            MissingField(field="name", message="Нет названия товара: без него карточку не собрать.")
        )
    if not result.filled_specs and not product.description:
        result.blocking = True
        result.missing.append(
            MissingField(
                field="specs, description",
                message="Нет ни характеристик, ни описания: фактов для текста нет.",
            )
        )

    if not product.manufacturer:
        result.missing.append(
            MissingField(field="manufacturer", message="Не указан производитель.")
        )
    if not product.description:
        result.missing.append(
            MissingField(
                field="description", message="Нет описания: назначение прибора неизвестно."
            )
        )
    elif len(product.description) < MIN_DESCRIPTION_CHARS:
        result.missing.append(
            MissingField(
                field="description",
                message="Описание очень короткое: текст может получиться общим.",
            )
        )

    if product.description.rstrip().endswith(("...", "…")):
        result.missing.append(
            MissingField(
                field="description",
                message="Описание обрывается многоточием: возможно, текст неполный.",
            )
        )

    for key, value in product.specs.items():
        if is_placeholder(value):
            result.missing.append(
                MissingField(field=f"specs.{key}", message="Значение пустое, нужно заполнить.")
            )

    for spec in expected_specs(product):
        if not any(spec.matches(key) for key in result.filled_specs):
            result.gaps.append(spec)
    return result


def build_characteristics(
    product: ProductInput,
    completeness: Completeness,
    extracted: list[Characteristic],
) -> tuple[list[Characteristic], list[MissingField]]:
    """Build the spec list by code from the input. Its values never pass through the LLM."""
    items = [
        Characteristic(
            name="производитель",
            value=product.manufacturer or NO_DATA,
            source="specs" if product.manufacturer else "none",
        )
    ]
    items += [
        Characteristic(
            name=key,
            value=value if not is_placeholder(value) else NO_DATA,
            source="specs" if not is_placeholder(value) else "none",
        )
        for key, value in product.specs.items()
    ]

    taken = {item.name.casefold() for item in items}
    accepted = [item for item in extracted if item.name.casefold() not in taken]
    items += accepted

    missing: list[MissingField] = []
    for spec in completeness.gaps:
        # An empty spec already has its row and its message, so it isn't added twice.
        if any(spec.matches(item.name) for item in items):
            continue
        items.append(Characteristic(name=spec.label, value=NO_DATA, source="none"))
        missing.append(
            MissingField(field=spec.label, message="Нет во входных данных, нужно уточнить.")
        )
    return items, missing
