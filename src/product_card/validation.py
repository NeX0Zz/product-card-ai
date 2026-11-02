"""Check an LLM draft against the requirements. Code decides here, not the model."""

import re

from .completeness import is_placeholder
from .grounding import SourceFacts, find_unsupported, quote_supports
from .models import (
    DESCRIPTION_MAX_CHARS,
    DESCRIPTION_MIN_CHARS,
    NO_DATA,
    QUERIES_MAX,
    QUERIES_MIN,
    TITLE_MAX_CHARS,
    Characteristic,
    Draft,
    Issue,
    ProductInput,
)

MAX_QUERY_CHARS = 100
MAX_COMMERCIAL_ITEM_CHARS = 200
_MARKUP = re.compile(r"(^|\n)\s*([#>*-]\s|\d+\.\s)|\*\*|`|<[a-zA-Z/]")


def _error(code: str, field: str, message: str, token: str | None = None) -> Issue:
    return Issue(severity="error", code=code, field=field, message=message, token=token)


def _check_title(title: str) -> list[Issue]:
    length = len(title)
    if not length:
        return [_error("title_empty", "seo_title", "Заголовок пуст.")]
    if length > TITLE_MAX_CHARS:
        excess = length - TITLE_MAX_CHARS
        return [
            _error(
                "title_too_long",
                "seo_title",
                f"В заголовке {length} символов, максимум {TITLE_MAX_CHARS}: "
                f"сократите минимум на {excess}.",
            )
        ]
    return []


def _check_description(text: str) -> list[Issue]:
    issues = []
    length = len(text)
    if length < DESCRIPTION_MIN_CHARS:
        issues.append(
            _error(
                "description_too_short",
                "seo_description",
                f"В описании {length} символов, минимум {DESCRIPTION_MIN_CHARS}: "
                f"добавьте минимум {DESCRIPTION_MIN_CHARS - length}. Для каждого параметра из "
                "данных допишите 1–2 предложения о том, зачем его сверять с задачей покупателя, "
                "без новых фактов.",
            )
        )
    elif length > DESCRIPTION_MAX_CHARS:
        issues.append(
            _error(
                "description_too_long",
                "seo_description",
                f"В описании {length} символов, максимум {DESCRIPTION_MAX_CHARS}: "
                f"сократите минимум на {length - DESCRIPTION_MAX_CHARS}.",
            )
        )
    if _MARKUP.search(text):
        issues.append(
            _error(
                "description_markup",
                "seo_description",
                "Описание должно быть обычным текстом: уберите разметку.",
            )
        )
    return issues


def _check_queries(queries: list[str]) -> list[Issue]:
    issues = []
    unique = {query.strip().casefold() for query in queries if query.strip()}
    if not QUERIES_MIN <= len(unique) <= QUERIES_MAX:
        issues.append(
            _error(
                "queries_count",
                "search_queries",
                f"Разных запросов {len(unique)}, нужно {QUERIES_MIN}–{QUERIES_MAX}.",
            )
        )
    if any(len(query) > MAX_QUERY_CHARS for query in queries):
        issues.append(
            _error(
                "query_too_long",
                "search_queries",
                f"Запрос длиннее {MAX_QUERY_CHARS} символов: сократите.",
            )
        )
    return issues


def _check_commercial(draft: Draft) -> list[Issue]:
    issues = []
    for field, items in (
        ("commercial.suitable_for", draft.commercial.suitable_for),
        ("commercial.solves", draft.commercial.solves),
    ):
        if not any(item.strip() for item in items):
            issues.append(_error("commercial_empty", field, "Блок пуст: добавьте 2–4 пункта."))
        elif any(len(item) > MAX_COMMERCIAL_ITEM_CHARS for item in items):
            issues.append(_error("commercial_too_long", field, "Пункт слишком длинный: сократите."))
    return issues


def _generated_texts(draft: Draft) -> list[tuple[str, str]]:
    return [
        ("seo_title", draft.seo_title),
        ("seo_description", draft.seo_description),
        *[(f"search_queries[{i}]", query) for i, query in enumerate(draft.search_queries)],
        *[
            (f"commercial.suitable_for[{i}]", t)
            for i, t in enumerate(draft.commercial.suitable_for)
        ],
        *[(f"commercial.solves[{i}]", t) for i, t in enumerate(draft.commercial.solves)],
    ]


def _check_grounding(draft: Draft, facts: SourceFacts) -> list[Issue]:
    issues = []
    for field, text in _generated_texts(draft):
        is_query = field.startswith("search_queries")
        for violation in find_unsupported(text, facts, is_query=is_query):
            issues.append(
                _error(f"unsupported_{violation.kind}", field, violation.message, violation.token)
            )
        if NO_DATA in text.casefold():
            issues.append(
                _error(
                    "no_data_in_text",
                    field,
                    f"Фраза «{NO_DATA}» допустима только в списке характеристик.",
                )
            )
    return issues


def validate_draft(draft: Draft, facts: SourceFacts) -> list[Issue]:
    return [
        *_check_title(draft.seo_title),
        *_check_description(draft.seo_description),
        *_check_queries(draft.search_queries),
        *_check_commercial(draft),
        *_check_grounding(draft, facts),
    ]


def accept_extracted(
    draft: Draft, product: ProductInput
) -> tuple[list[Characteristic], list[Issue]]:
    """Keep a spec extracted from the description only if its quote is really there."""
    accepted: list[Characteristic] = []
    issues: list[Issue] = []
    for item in draft.extracted_specs:
        name, value = item.name.strip(), item.value.strip()
        if not name or is_placeholder(value):
            continue
        if quote_supports(item.quote, value, product.description, name):
            accepted.append(
                Characteristic(
                    name=name, value=value, source="description", quote=item.quote.strip()
                )
            )
        else:
            issues.append(
                Issue(
                    severity="warning",
                    code="extracted_spec_rejected",
                    field=name,
                    message="Значение отброшено: модель не подтвердила его цитатой из описания.",
                )
            )
    return accepted, issues
