"""Export a card to Markdown and plain text for copying into a CMS."""

import re

from .models import ProductCard

_STATUS_LABELS = {
    "ready": "готово к проверке",
    "needs_clarification": "есть что уточнить",
    "needs_review": "есть ошибки, нужна правка",
    "insufficient_data": "недостаточно данных",
}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|])")


def _md(text: str) -> str:
    """Escape input-derived text, so product data cannot inject markup or links."""
    return _MD_SPECIAL.sub(r"\\\1", text).replace("\n", " ")


def _attention_lines(card: ProductCard) -> list[str]:
    lines = [f"{item.field}: {item.message}" for item in card.missing_fields]
    lines += [f"{item.field}: {item.question}" for item in card.clarifications]
    lines += [f"{item.field}: {item.message}" for item in card.issues]
    return lines


def to_markdown(card: ProductCard) -> str:
    title = _md(card.product_name) or "товар без названия"
    parts = [f"# {title}", f"**Статус:** {_STATUS_LABELS[card.status]}"]
    if card.seo_title:
        parts += ["## SEO-заголовок", _md(card.seo_title)]
    if card.seo_description:
        parts += ["## SEO-описание", _md(card.seo_description)]
    if card.characteristics:
        rows = "\n".join(f"| {_md(c.name)} | {_md(c.value)} |" for c in card.characteristics)
        parts += ["## Ключевые характеристики", f"| Параметр | Значение |\n|---|---|\n{rows}"]
    if card.search_queries:
        parts += ["## Поисковые запросы", "\n".join(f"- {_md(q)}" for q in card.search_queries)]
    if card.commercial.suitable_for or card.commercial.solves:
        parts.append("## Блок для менеджера")
        parts += [
            "**Кому подходит:**",
            "\n".join(f"- {_md(i)}" for i in card.commercial.suitable_for),
        ]
        parts += [
            "**Какие задачи решает:**",
            "\n".join(f"- {_md(i)}" for i in card.commercial.solves),
        ]
    attention = _attention_lines(card)
    if attention:
        parts += ["## Требует внимания", "\n".join(f"- {_md(line)}" for line in attention)]
    return "\n\n".join(parts) + "\n"


def to_text(card: ProductCard) -> str:
    parts = [card.product_name or "Товар без названия", f"Статус: {_STATUS_LABELS[card.status]}"]
    if card.seo_title:
        parts.append(f"SEO-заголовок:\n{card.seo_title}")
    if card.seo_description:
        parts.append(f"SEO-описание:\n{card.seo_description}")
    if card.characteristics:
        rows = "\n".join(f"{c.name}: {c.value}" for c in card.characteristics)
        parts.append(f"Ключевые характеристики:\n{rows}")
    if card.search_queries:
        parts.append("Поисковые запросы:\n" + "\n".join(card.search_queries))
    if card.commercial.suitable_for or card.commercial.solves:
        parts.append("Кому подходит:\n" + "\n".join(card.commercial.suitable_for))
        parts.append("Какие задачи решает:\n" + "\n".join(card.commercial.solves))
    attention = _attention_lines(card)
    if attention:
        parts.append("Требует внимания:\n" + "\n".join(attention))
    return "\n\n".join(parts) + "\n"
