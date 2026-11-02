"""Generation pipeline: check input, call LLM, validate, repair, assemble the card."""

import asyncio
import logging
import time

from pydantic import ValidationError

from .completeness import build_characteristics, check_completeness
from .config import Settings
from .grounding import build_source_facts, find_conflicts
from .llm import LLMBadOutput, LLMClient, LLMError, LLMResult
from .models import (
    CardMeta,
    CardStatus,
    Clarification,
    Draft,
    Issue,
    MissingField,
    ProductCard,
    ProductInput,
)
from .prompts import PROMPT_VERSION, build_messages, build_repair_messages, parse_draft
from .validation import accept_extracted, validate_draft

logger = logging.getLogger(__name__)

_MAX_CLARIFICATIONS = 10
# Models tend to ask whether a brand is misspelled. Gaps in these fields are found by code.
_UNASKED_FIELDS = {"name", "manufacturer", "description", "название", "производитель", "описание"}
_INVALID_OUTPUT_ISSUE = Issue(
    severity="error",
    code="invalid_output",
    field="response",
    message="Ответ не соответствует JSON-схеме: верни все поля в требуемом формате.",
)


def _add_usage(meta: CardMeta, result: LLMResult) -> None:
    meta.models_used.append(result.model)
    meta.prompt_tokens += result.prompt_tokens
    meta.completion_tokens += result.completion_tokens
    if result.cost_usd is not None:
        meta.cost_usd = (meta.cost_usd or 0.0) + result.cost_usd


def _ignored_field_notes(ignored: list[str]) -> list[MissingField]:
    if not ignored:
        return []
    names = ", ".join(ignored)
    return [MissingField(field=names, message="Неизвестные поля проигнорированы.")]


def _deadline_error() -> LLMError:
    return LLMError("llm_timeout", "Генерация заняла слишком много времени. Повторите запрос.", 504)


def _select_models(attempt: int, settings: Settings) -> list[str]:
    """Cheap chain first. The last repair goes to a stronger chain, if there was a failure."""
    is_last_repair = attempt == settings.llm_max_attempts and attempt > 1
    return settings.escalation_models if is_last_repair else settings.models


def _conflict_clarifications(product: ProductInput) -> list[Clarification]:
    """Code finds these itself, so they don't depend on the model noticing the conflict."""
    return [
        Clarification(
            field=conflict.field,
            question=(
                f"В характеристиках указано «{conflict.spec_value}», а в описании рядом с этим "
                f"параметром — «{conflict.description_value}». Какое значение верное?"
            ),
        )
        for conflict in find_conflicts(product)
    ]


def precheck_card(
    product: ProductInput, ignored_fields: list[str] | None = None
) -> ProductCard | None:
    """Return a card if the data is too thin to generate from, without calling the LLM."""
    completeness = check_completeness(product)
    if not completeness.blocking:
        return None
    return ProductCard(
        status="insufficient_data",
        product_name=product.name,
        missing_fields=[*completeness.missing, *_ignored_field_notes(ignored_fields or [])],
        meta=CardMeta(prompt_version=PROMPT_VERSION),
    )


async def generate_card(
    product: ProductInput,
    llm: LLMClient,
    settings: Settings,
    ignored_fields: list[str] | None = None,
) -> ProductCard:
    started = time.perf_counter()
    meta = CardMeta(prompt_version=PROMPT_VERSION)
    insufficient = precheck_card(product, ignored_fields)
    if insufficient is not None:
        return insufficient

    completeness = check_completeness(product)
    notes = _ignored_field_notes(ignored_fields or [])
    facts = build_source_facts(product)
    messages = build_messages(product)
    best: tuple[Draft, list[Issue]] | None = None
    last_error: LLMError | None = None

    for attempt in range(1, settings.llm_max_attempts + 1):
        remaining = settings.generation_deadline_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            last_error = last_error or _deadline_error()
            break
        meta.attempts = attempt
        try:
            async with asyncio.timeout(remaining):
                result = await llm.complete(messages, models=_select_models(attempt, settings))
        except TimeoutError as exc:
            # Same rule as for API failures below: a draft that already exists is kept.
            if best is None:
                raise _deadline_error() from exc
            last_error = _deadline_error()
            break
        except LLMBadOutput as exc:
            last_error = exc
            continue
        except LLMError as exc:
            # An API failure on a repair must not throw away a draft that already exists.
            if best is None:
                raise
            last_error = exc
            break
        _add_usage(meta, result)

        try:
            draft = parse_draft(result.content)
        except ValidationError:
            last_error = LLMBadOutput()
            issues = [_INVALID_OUTPUT_ISSUE]
            messages = build_repair_messages(product, result.content[:8000], issues)
            continue

        issues = validate_draft(draft, facts)
        if best is None or len(issues) < len(best[1]):
            best = (draft, issues)
        if not issues:
            break
        logger.info("attempt=%s validation_errors=%s", attempt, [i.code for i in issues])
        messages = build_repair_messages(product, result.content, issues)

    meta.duration_ms = int((time.perf_counter() - started) * 1000)
    if best is None:
        raise last_error or LLMBadOutput()

    draft, issues = best
    extracted, extraction_issues = accept_extracted(draft, product)
    characteristics, gaps = build_characteristics(product, completeness, extracted)
    missing = [*completeness.missing, *gaps, *notes]
    code_found = _conflict_clarifications(product)
    code_fields = {item.field.casefold() for item in code_found} | _UNASKED_FIELDS
    from_model = [item for item in draft.clarifications if item.field.casefold() not in code_fields]
    clarifications = [*code_found, *from_model][:_MAX_CLARIFICATIONS]
    all_issues = [*issues, *extraction_issues]

    status: CardStatus
    if any(issue.severity == "error" for issue in all_issues):
        status = "needs_review"
    elif missing or clarifications:
        status = "needs_clarification"
    else:
        status = "ready"

    return ProductCard(
        status=status,
        product_name=product.name,
        seo_title=draft.seo_title,
        seo_description=draft.seo_description,
        characteristics=characteristics,
        # Search engines ignore case, and a lowercase list is easier to compare and paste.
        search_queries=[query.strip().lower() for query in draft.search_queries if query.strip()],
        commercial=draft.commercial,
        missing_fields=missing,
        clarifications=clarifications,
        issues=all_issues,
        meta=meta,
    )
