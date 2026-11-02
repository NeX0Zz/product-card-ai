"""Data contracts: raw input parsing, the LLM draft and the final product card."""

import json
import math
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_NAME_CHARS = 300
MAX_MANUFACTURER_CHARS = 200
MAX_DESCRIPTION_CHARS = 20_000
MAX_SPECS = 100
MAX_SPEC_KEY_CHARS = 200
MAX_SPEC_VALUE_CHARS = 1_000

TITLE_MAX_CHARS = 70
DESCRIPTION_MIN_CHARS = 800
DESCRIPTION_MAX_CHARS = 1_200
QUERIES_MIN = 3
QUERIES_MAX = 5

NO_DATA = "данных нет"

KNOWN_FIELDS = {"name", "manufacturer", "specs", "description"}


class FieldError(BaseModel):
    field: str
    message: str


class InputError(Exception):
    """Input cannot be used. Its messages are safe to show to the user."""

    def __init__(self, code: str, message: str, details: list[FieldError] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []


class ProductInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = ""
    manufacturer: str = ""
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""


def clean_text(value: str) -> str:
    """Drop control and invisible format characters, which can hide injected instructions."""
    kept = (
        char
        for char in value
        if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Co"}
    )
    return "".join(kept).strip()


def _scalar_to_text(value: Any) -> str | None:
    """Convert a JSON scalar to text. Containers are not scalars and return None."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, int | float):
        return str(value)
    return None


def _text_field(raw: dict[str, Any], name: str, limit: int, errors: list[FieldError]) -> str:
    text = _scalar_to_text(raw.get(name))
    if text is None:
        errors.append(FieldError(field=name, message="Ожидается строка."))
        return ""
    if len(text) > limit:
        errors.append(FieldError(field=name, message=f"Длина больше {limit} символов."))
        return ""
    return text


def _specs_field(raw: dict[str, Any], errors: list[FieldError]) -> dict[str, str]:
    value = raw.get("specs")
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(FieldError(field="specs", message="Ожидается объект «название: значение»."))
        return {}
    if len(value) > MAX_SPECS:
        errors.append(FieldError(field="specs", message=f"Больше {MAX_SPECS} характеристик."))
        return {}

    specs: dict[str, str] = {}
    for key, item in value.items():
        clean_key = clean_text(key)
        field = f"specs.{clean_key[:60]}"
        text = _scalar_to_text(item)
        if not clean_key:
            errors.append(FieldError(field="specs", message="Пустое название характеристики."))
        elif len(clean_key) > MAX_SPEC_KEY_CHARS:
            errors.append(FieldError(field=field, message="Слишком длинное название."))
        elif clean_key in specs:
            errors.append(FieldError(field=field, message="Название повторяется."))
        elif text is None:
            errors.append(FieldError(field=field, message="Значение должно быть строкой."))
        elif len(text) > MAX_SPEC_VALUE_CHARS:
            errors.append(FieldError(field=field, message="Слишком длинное значение."))
        else:
            specs[clean_key] = text
    return specs


def parse_product(raw: Any) -> tuple[ProductInput, list[str]]:
    """Validate decoded JSON. Returns the product and the names of ignored fields."""
    if not isinstance(raw, dict):
        raise InputError("invalid_product", "Ожидается JSON-объект с данными одного товара.")

    errors: list[FieldError] = []
    product = ProductInput(
        name=_text_field(raw, "name", MAX_NAME_CHARS, errors),
        manufacturer=_text_field(raw, "manufacturer", MAX_MANUFACTURER_CHARS, errors),
        specs=_specs_field(raw, errors),
        description=_text_field(raw, "description", MAX_DESCRIPTION_CHARS, errors),
    )
    if errors:
        raise InputError("invalid_product", "Данные товара заполнены некорректно.", errors)
    ignored = sorted(clean_text(str(key))[:60] for key in raw if key not in KNOWN_FIELDS)
    return product, ignored


def _reject_constant(token: str) -> None:
    raise ValueError(f"недопустимое значение {token}")


def load_json(text: str) -> Any:
    """Decode user JSON. Errors name the position and never echo the content."""
    if not text.strip():
        raise InputError("invalid_json", "Пустой запрос: передайте JSON с данными товара.")
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise InputError(
            "invalid_json",
            f"Некорректный JSON: строка {exc.lineno}, позиция {exc.colno}.",
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise InputError("invalid_json", "Некорректный JSON: недопустимое значение.") from exc


class DraftCommercial(BaseModel):
    suitable_for: list[str] = Field(default_factory=list)
    solves: list[str] = Field(default_factory=list)


class ExtractedSpec(BaseModel):
    name: str
    value: str
    quote: str


class Clarification(BaseModel):
    field: str
    question: str


class Draft(BaseModel):
    """Raw LLM answer. It is validated by code before anything reaches the user."""

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial: DraftCommercial
    extracted_specs: list[ExtractedSpec] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)

    @field_validator("seo_title", "seo_description")
    @classmethod
    def _clean(cls, value: str) -> str:
        """Lengths are limits on what gets published, so measure the cleaned text."""
        return clean_text(value)


class Characteristic(BaseModel):
    name: str
    value: str
    source: Literal["specs", "description", "none"]
    quote: str | None = None


class MissingField(BaseModel):
    field: str
    message: str


class Issue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    field: str
    message: str
    token: str | None = None


class CardMeta(BaseModel):
    models_used: list[str] = Field(default_factory=list)
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    duration_ms: int = 0
    cached: bool = False
    prompt_version: str = ""


CardStatus = Literal["ready", "needs_clarification", "needs_review", "insufficient_data"]


class ProductCard(BaseModel):
    """Final result.

    ready: passed validation, nothing to clarify.
    needs_clarification: passed validation, but some data is missing or questionable.
    needs_review: validation errors remain after all attempts, see `issues`.
    insufficient_data: too little input, the LLM was not called.
    """

    status: CardStatus
    product_name: str = ""
    seo_title: str = ""
    seo_description: str = ""
    characteristics: list[Characteristic] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    commercial: DraftCommercial = Field(default_factory=DraftCommercial)
    missing_fields: list[MissingField] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    meta: CardMeta = Field(default_factory=CardMeta)
