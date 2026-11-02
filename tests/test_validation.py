import pytest

from product_card.grounding import SourceFacts, build_source_facts
from product_card.models import Draft, Issue
from product_card.validation import accept_extracted, validate_draft

from .fakes import dm02_product, draft_payload


@pytest.fixture
def facts() -> SourceFacts:
    return build_source_facts(dm02_product())


def draft_with(**overrides: object) -> Draft:
    return Draft.model_validate(draft_payload(**overrides))


def codes(issues: list[Issue]) -> list[str]:
    return [issue.code for issue in issues]


def issue_with_code(issues: list[Issue], code: str) -> Issue:
    return next(issue for issue in issues if issue.code == code)


def queries(count: int) -> list[str]:
    names = ["альфа", "бета", "гамма", "дельта", "эпсилон", "дзета", "эта"]
    return [f"манометр {name}" for name in names[:count]]


def test_valid_draft_has_no_issues(facts: SourceFacts) -> None:
    assert validate_draft(draft_with(), facts) == []


@pytest.mark.parametrize(("length", "valid"), [(1, True), (69, True), (70, True), (71, False)])
def test_title_length_boundary(length: int, valid: bool, facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(seo_title="а" * length), facts)

    assert ("title_too_long" not in codes(issues)) is valid


def test_empty_title_is_an_error(facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(seo_title=""), facts)

    assert "title_empty" in codes(issues)


def test_long_title_message_tells_how_much_to_cut(facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(seo_title="а" * 71), facts)

    message = issue_with_code(issues, "title_too_long").message
    assert "71" in message
    assert "сократите минимум на 1" in message


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (799, "description_too_short"),
        (800, None),
        (1200, None),
        (1201, "description_too_long"),
    ],
)
def test_description_length_boundary(length: int, expected: str | None, facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(seo_description="а" * length), facts)

    length_codes = {"description_too_short", "description_too_long"} & set(codes(issues))
    assert length_codes == ({expected} if expected else set())


def test_short_description_message_tells_how_much_to_add(facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(seo_description="а" * 750), facts)

    message = issue_with_code(issues, "description_too_short").message
    assert "750" in message
    assert "добавьте минимум 50" in message


def test_long_description_message_tells_how_much_to_cut(facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(seo_description="а" * 1260), facts)

    message = issue_with_code(issues, "description_too_long").message
    assert "сократите минимум на 60" in message


@pytest.mark.parametrize(
    ("count", "valid"), [(0, False), (1, False), (2, False), (3, True), (5, True), (6, False)]
)
def test_search_query_count_boundary(count: int, valid: bool, facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(search_queries=queries(count)), facts)

    assert ("queries_count" not in codes(issues)) is valid


def test_duplicate_queries_are_counted_once(facts: SourceFacts) -> None:
    duplicated = ["манометр альфа", "Манометр Альфа", " манометр альфа ", "манометр бета"]

    issues = validate_draft(draft_with(search_queries=duplicated), facts)

    assert "Разных запросов 2" in issue_with_code(issues, "queries_count").message


def test_blank_queries_are_not_counted(facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(search_queries=[*queries(2), "  ", ""]), facts)

    assert "queries_count" in codes(issues)


def test_overlong_query_is_an_error(facts: SourceFacts) -> None:
    long_query = "манометр " + "а" * 100

    issues = validate_draft(draft_with(search_queries=[*queries(2), long_query]), facts)

    assert "query_too_long" in codes(issues)


@pytest.mark.parametrize("block", ["suitable_for", "solves"])
@pytest.mark.parametrize("empty", [[], [""], ["  "]])
def test_empty_commercial_list_is_an_error(
    block: str, empty: list[str], facts: SourceFacts
) -> None:
    commercial = {"suitable_for": ["Служба КИП"], "solves": ["Контроль давления"], block: empty}

    issues = validate_draft(draft_with(commercial=commercial), facts)

    assert issue_with_code(issues, "commercial_empty").field == f"commercial.{block}"


def test_overlong_commercial_item_is_an_error(facts: SourceFacts) -> None:
    commercial = {"suitable_for": ["а" * 201], "solves": ["Контроль давления"]}

    issues = validate_draft(draft_with(commercial=commercial), facts)

    assert "commercial_too_long" in codes(issues)


@pytest.mark.parametrize(
    "markup",
    ["**важно**", "# Заголовок", "- пункт списка", "1. пункт", "> цитата", "`код`", "<b>текст</b>"],
)
def test_markup_in_description_is_an_error(markup: str, facts: SourceFacts) -> None:
    description = "а" * 400 + "\n" + markup + "\n" + "а" * 400

    issues = validate_draft(draft_with(seo_description=description), facts)

    assert "description_markup" in codes(issues)


def test_plain_description_has_no_markup_issue(facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(), facts)

    assert "description_markup" not in codes(issues)


@pytest.mark.parametrize(
    "field",
    [
        "seo_title",
        "seo_description",
        "search_queries",
        "commercial_suitable_for",
        "commercial_solves",
    ],
)
def test_no_data_phrase_is_rejected_in_every_seo_text(field: str, facts: SourceFacts) -> None:
    payload = draft_payload()
    if field == "seo_title":
        payload["seo_title"] = "Манометр, данных нет"
    elif field == "seo_description":
        payload["seo_description"] = str(payload["seo_description"])[:-20] + " Данных нет. "
    elif field == "search_queries":
        payload["search_queries"] = ["манометр данных нет", "манометр альфа", "манометр бета"]
    elif field == "commercial_suitable_for":
        payload["commercial"] = {"suitable_for": ["Данных нет"], "solves": ["Контроль"]}
    else:
        payload["commercial"] = {"suitable_for": ["Служба КИП"], "solves": ["данных нет"]}

    issues = validate_draft(Draft.model_validate(payload), facts)

    assert "no_data_in_text" in codes(issues)


def test_invented_value_in_title_is_reported_with_field_and_token(facts: SourceFacts) -> None:
    issues = validate_draft(draft_with(seo_title="Манометр DM 02, 0–2,5 МПа"), facts)

    issue = issue_with_code(issues, "unsupported_number")
    assert issue.field == "seo_title"
    assert issue.token == "2,5 МПа"
    assert issue.severity == "error"


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        (
            "search_queries[0]",
            {"search_queries": ["манометр IP65", "манометр альфа", "манометр бета"]},
        ),
        (
            "commercial.suitable_for[0]",
            {"commercial": {"suitable_for": ["Для RS-485"], "solves": ["Контроль"]}},
        ),
        (
            "commercial.solves[0]",
            {"commercial": {"suitable_for": ["Служба КИП"], "solves": ["Поверка на 2 года"]}},
        ),
    ],
)
def test_grounding_checks_queries_and_commercial_blocks(
    field: str, overrides: dict[str, object], facts: SourceFacts
) -> None:
    issues = validate_draft(draft_with(**overrides), facts)

    assert field in {issue.field for issue in issues if issue.code.startswith("unsupported_")}


def test_extracted_spec_with_real_quote_is_accepted() -> None:
    product = dm02_product(description="Напряжение питания 24 В постоянного тока. Прибор надёжен.")
    draft = draft_with(
        extracted_specs=[
            {"name": "питание", "value": "24 В", "quote": "Напряжение питания 24 В"},
        ]
    )

    accepted, issues = accept_extracted(draft, product)

    assert issues == []
    assert [(c.name, c.value, c.source, c.quote) for c in accepted] == [
        ("питание", "24 В", "description", "Напряжение питания 24 В")
    ]


def test_extracted_spec_with_fabricated_quote_gives_a_warning() -> None:
    draft = draft_with(
        extracted_specs=[{"name": "питание", "value": "24 В", "quote": "Питание 24 В от сети"}]
    )

    accepted, issues = accept_extracted(draft, dm02_product())

    assert accepted == []
    assert [(i.severity, i.code, i.field) for i in issues] == [
        ("warning", "extracted_spec_rejected", "питание")
    ]


@pytest.mark.parametrize("value", ["", "н/д", "уточняется", "-"])
def test_extracted_spec_with_placeholder_value_is_skipped_silently(value: str) -> None:
    draft = draft_with(extracted_specs=[{"name": "питание", "value": value, "quote": "x"}])

    assert accept_extracted(draft, dm02_product()) == ([], [])


def test_extracted_spec_without_name_is_skipped_silently() -> None:
    draft = draft_with(extracted_specs=[{"name": "  ", "value": "24 В", "quote": "24 В"}])

    assert accept_extracted(draft, dm02_product()) == ([], [])
