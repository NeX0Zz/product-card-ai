import re

import pytest

from product_card.models import (
    NO_DATA,
    Characteristic,
    Clarification,
    DraftCommercial,
    Issue,
    MissingField,
    ProductCard,
)
from product_card.render import to_markdown, to_text

ATTENTION = "Требует внимания"


def card(**overrides: object) -> ProductCard:
    fields: dict[str, object] = {
        "status": "ready",
        "product_name": "Манометр DM 02",
        "seo_title": "Манометр DM 02",
        "seo_description": "Описание прибора.",
        "characteristics": [Characteristic(name="класс точности", value="1,5", source="specs")],
        "search_queries": ["манометр dm 02"],
        "commercial": DraftCommercial(suitable_for=["Служба КИП"], solves=["Контроль давления"]),
    }
    fields.update(overrides)
    return ProductCard(**fields)  # type: ignore[arg-type]


def test_markdown_contains_all_sections_of_a_ready_card() -> None:
    markdown = to_markdown(card())

    for heading in (
        "# Манометр DM 02",
        "## SEO-заголовок",
        "## SEO-описание",
        "## Ключевые характеристики",
        "## Поисковые запросы",
        "## Блок для менеджера",
    ):
        assert heading in markdown
    assert "| класс точности | 1,5 |" in markdown
    assert "- Служба КИП" in markdown


def test_pipe_in_product_data_cannot_break_the_table() -> None:
    markdown = to_markdown(
        card(characteristics=[Characteristic(name="a|b", value="c|d", source="specs")])
    )

    assert "| a\\|b | c\\|d |" in markdown


def test_script_tag_is_escaped() -> None:
    markdown = to_markdown(
        card(product_name="<script>alert(1)</script>", seo_title="<img src=x onerror=alert(1)>")
    )

    assert not re.search(r"(?<!\\)<", markdown)
    assert "\\<script\\>" in markdown


def test_markdown_link_from_product_data_is_neutralised() -> None:
    markdown = to_markdown(
        card(
            product_name="[click](http://evil.example)",
            characteristics=[
                Characteristic(
                    name="[x](http://evil.example)", value="![i](http://e/p.png)", source="specs"
                )
            ],
        )
    )

    assert "[click](" not in markdown
    assert "[x](" not in markdown
    assert "![i](" not in markdown
    assert "\\[click\\]\\(http://evil.example\\)" in markdown


def test_newline_in_data_cannot_start_a_new_markdown_block() -> None:
    markdown = to_markdown(card(product_name="Манометр\n# Взломан\n- пункт"))

    assert "\n# Взломан" not in markdown
    assert "\n- пункт" not in markdown


def test_missing_values_are_shown_as_no_data() -> None:
    gap = Characteristic(name="питание", value=NO_DATA, source="none")

    markdown = to_markdown(card(characteristics=[gap]))
    text = to_text(card(characteristics=[gap]))

    assert f"| питание | {NO_DATA} |" in markdown
    assert f"питание: {NO_DATA}" in text


def test_unnamed_product_gets_a_placeholder_title() -> None:
    insufficient = card(
        status="insufficient_data", product_name="", seo_title="", seo_description=""
    )

    assert "# товар без названия" in to_markdown(insufficient)
    assert to_text(insufficient).startswith("Товар без названия")


def test_attention_section_is_absent_when_nothing_needs_attention() -> None:
    ready = card()

    assert ATTENTION not in to_markdown(ready)
    assert ATTENTION not in to_text(ready)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"missing_fields": [MissingField(field="питание", message="Нет данных.")]},
            "питание: Нет данных.",
        ),
        (
            {"clarifications": [Clarification(field="диапазон", question="Какой верный?")]},
            "диапазон: Какой верный?",
        ),
        (
            {
                "issues": [
                    Issue(severity="error", code="x", field="диапазон", message="Слишком длинный.")
                ]
            },
            "диапазон: Слишком длинный.",
        ),
    ],
)
def test_attention_section_lists_each_kind_of_problem(
    overrides: dict[str, object], expected: str
) -> None:
    problem_card = card(status="needs_clarification", **overrides)

    assert ATTENTION in to_markdown(problem_card)
    assert expected in to_markdown(problem_card)
    assert ATTENTION in to_text(problem_card)
    assert expected in to_text(problem_card)


def test_attention_lines_are_escaped_in_markdown() -> None:
    problem_card = card(missing_fields=[MissingField(field="<b>x</b>", message="[a](http://e)")])

    markdown = to_markdown(problem_card)

    assert not re.search(r"(?<!\\)<", markdown)
    assert "[a](" not in markdown


def test_status_is_shown_in_human_words() -> None:
    assert "готово к проверке" in to_markdown(card())
    assert "недостаточно данных" in to_text(card(status="insufficient_data"))


def test_plain_text_export_keeps_data_verbatim() -> None:
    text = to_text(card(product_name="Манометр | <DM 02>"))

    assert text.startswith("Манометр | <DM 02>")


def test_empty_commercial_block_is_omitted() -> None:
    bare = card(commercial=DraftCommercial())

    assert "Блок для менеджера" not in to_markdown(bare)
    assert "Кому подходит" not in to_text(bare)
