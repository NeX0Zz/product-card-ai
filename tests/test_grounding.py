import pytest

from product_card.grounding import (
    SourceFacts,
    build_source_facts,
    find_conflicts,
    find_unsupported,
    quote_supports,
)
from product_card.models import ProductInput

from .fakes import dm02_product


def facts_for(**fields: object) -> SourceFacts:
    return build_source_facts(ProductInput(**fields))  # type: ignore[arg-type]


def violation_kinds(text: str, facts: SourceFacts) -> set[str]:
    return {violation.kind for violation in find_unsupported(text, facts)}


@pytest.fixture
def dm02_facts() -> SourceFacts:
    return build_source_facts(dm02_product())


@pytest.mark.parametrize(
    "text",
    [
        "0–1,6 МПа",
        "0-1,6 МПа",
        "0 — 1,6 МПа",
        "0−1,6 МПа",
        "0 – 1,6 МПа",
        "0–1.6 МПа",
        "1.6 МПа",
        "от 0 до 1,6 МПа",
        "0 – 1,6 МПа",
        "0–1,6МПа",
        "0–1,6 мпа",
        "класс точности 1,5",
        "присоединение G1/2",
        "присоединение G 1/2",
        "диаметр корпуса 100 мм",
        "модель DM 02",
        "производитель Манотомь",
        "Прибор предназначен для измерения избыточного давления.",
        "Подходит для B2B",
        "Каталог B2B и розница",
        "",
    ],
)
def test_text_backed_by_source_is_allowed(text: str, dm02_facts: SourceFacts) -> None:
    assert find_unsupported(text, dm02_facts) == []


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("0–2,5 МПа", "number"),
        ("класс точности 1,7", "number"),
        ("16 бар", "number"),
        ("1,6 бар", "unit"),
        ("1,6 кПа", "unit"),
        ("1,5 %", "unit"),
        ("100 см", "unit"),
        ("присоединение G1/4", "code"),
        ("присоединение M20x1,5", "code"),
        ("степень защиты IP65", "code"),
        ("интерфейс RS-485", "code"),
        ("по ГОСТ 2405-88", "code"),
        ("корпус из нержавеющей стали", "claim"),
        ("латунный штуцер", "claim"),
        ("есть поверка", "claim"),
        ("гарантия 2 года", "claim"),
        ("поддерживает Modbus", "claim"),
        ("высокая точность", "claim"),
        ("высокоточный прибор", "claim"),
        ("надежный прибор", "claim"),
        ("лучший манометр", "claim"),
        ("в наличии на складе", "claim"),
        ("выпускается в двух исполнениях", "claim"),
        ("https://example.com/dm02", "contact"),
        ("www.example.com", "contact"),
        ("sales@example.ru", "contact"),
        ("звоните +7 (495) 123-45-67", "contact"),
        ("с поддержкой Bluetooth", "latin"),
    ],
)
def test_text_not_backed_by_source_is_rejected(
    text: str, kind: str, dm02_facts: SourceFacts
) -> None:
    assert kind in violation_kinds(text, dm02_facts)


def test_converted_unit_is_reported_with_the_offending_value(dm02_facts: SourceFacts) -> None:
    violations = find_unsupported("Диапазон 16 бар", dm02_facts)

    assert [(v.kind, v.token) for v in violations] == [("number", "16 бар")]


def test_known_number_with_other_unit_is_a_unit_violation(dm02_facts: SourceFacts) -> None:
    violations = find_unsupported("Диапазон 1,6 бар", dm02_facts)

    assert [(v.kind, v.token) for v in violations] == [("unit", "1,6 бар")]


@pytest.mark.parametrize(
    ("text", "allowed"),
    [
        ("от -40 до +60 °C", True),
        ("-40…+60 °C", True),
        ("+60 °C", True),
        ("+40 °C", False),
        ("-60 °C", False),
    ],
)
def test_temperature_sign_is_part_of_the_value(text: str, allowed: bool) -> None:
    facts = facts_for(name="Термометр", specs={"диапазон": "-40…+60 °C"})

    assert (find_unsupported(text, facts) == []) is allowed


def test_flipping_sign_of_positive_source_temperature_is_rejected() -> None:
    facts = facts_for(name="Термометр", specs={"диапазон": "+40 °C"})

    assert find_unsupported("-40 °C", facts)


@pytest.mark.parametrize("source_unit", ["°C", "°С"])
@pytest.mark.parametrize("text_unit", ["°C", "°С"])
def test_cyrillic_and_latin_c_in_degrees_are_equivalent(source_unit: str, text_unit: str) -> None:
    facts = facts_for(name="Термометр", specs={"диапазон": f"от 0 до 150 {source_unit}"})

    assert find_unsupported(f"до 150 {text_unit}", facts) == []


def test_upper_bound_of_ellipsis_range_keeps_its_unit() -> None:
    facts = facts_for(name="Термометр", specs={"диапазон": "0…150 °C"})

    assert find_unsupported("до 150 °C", facts) == []


def test_homoglyphs_in_identifiers_do_not_hide_a_different_code() -> None:
    facts = facts_for(name="Датчик", specs={"защита": "IP54"})

    assert find_unsupported("защита IP54", facts) == []
    assert find_unsupported("защита IP65", facts)


def test_non_breaking_space_in_source_matches_plain_space_in_text() -> None:
    facts = facts_for(name="Манометр", specs={"диапазон": "0–1,6 МПа"})

    assert find_unsupported("0–1,6 МПа", facts) == []


def test_unit_in_spec_key_allows_value_with_that_unit() -> None:
    facts = facts_for(name="Манометр", specs={"диаметр, мм": "100"})

    assert find_unsupported("диаметр 100 мм", facts) == []
    assert find_unsupported("диаметр 100 см", facts)


def test_list_sharing_one_unit_allows_each_value_with_that_unit() -> None:
    facts = facts_for(name="Манометр", specs={"диапазоны": "0,6; 1,0; 1,6 МПа"})

    assert find_unsupported("0,6 МПа", facts) == []
    assert find_unsupported("1,0 МПа", facts) == []
    assert find_unsupported("0,6 бар", facts)


@pytest.mark.parametrize("text", ["100 с манометром", "100 в системе", "100 м от точки"])
def test_word_after_a_number_is_not_mistaken_for_a_unit(text: str) -> None:
    facts = facts_for(name="Манометр", specs={"количество": "100"})

    assert find_unsupported(text, facts) == []


def test_two_with_pressure_gauge_is_not_a_unit_violation() -> None:
    facts = facts_for(name="Комплект", specs={"состав": "2 манометра"})

    assert find_unsupported("комплект из 2 с манометром", facts) == []


def test_volt_from_source_is_allowed_and_preposition_v_is_not_a_unit() -> None:
    facts = facts_for(name="Датчик", specs={"питание": "24 В"})

    assert find_unsupported("питание 24 В", facts) == []
    assert find_unsupported("работает в 24 В сети", facts) == []
    assert find_unsupported("питание 12 В", facts)
    assert find_unsupported("питание 24 А", facts)


def test_text_without_any_facts_is_allowed(dm02_facts: SourceFacts) -> None:
    text = "Подходит для служб КИП и инженеров, которые подбирают прибор на замену."

    assert find_unsupported(text, dm02_facts) == []


def test_b2b_with_hyphenated_russian_suffix_is_allowed(dm02_facts: SourceFacts) -> None:
    assert find_unsupported("Подходит для B2B-закупок", dm02_facts) == []


def test_b2b_is_always_allowed() -> None:
    assert find_unsupported("Карточка для B2B", facts_for(name="Прибор")) == []


def test_claim_is_allowed_when_the_same_stem_is_in_source() -> None:
    facts = facts_for(name="Манометр", description="Корпус из нержавеющей стали.")

    assert "claim" not in violation_kinds("корпус из нержавеющей стали", facts)


def test_each_violation_is_reported_once(dm02_facts: SourceFacts) -> None:
    violations = find_unsupported("16 бар и снова 16 бар", dm02_facts)

    assert len(violations) == 1


def test_quote_supports_accepts_real_quote_with_the_value() -> None:
    description = "Прибор рассчитан на питание. Напряжение питания 24 В постоянного тока."

    assert quote_supports("Напряжение питания 24 В", "24 В", description)


def test_quote_supports_accepts_quote_with_other_dash_and_spacing() -> None:
    description = "Диапазон измерений 0 – 1,6 МПа."

    assert quote_supports("Диапазон измерений 0-1,6 МПа", "0–1,6 МПа", description)


def test_quote_supports_rejects_fabricated_quote() -> None:
    description = "Прибор рассчитан на питание. Напряжение питания 24 В."

    assert not quote_supports("Напряжение питания 36 В", "36 В", description)


def test_quote_supports_rejects_real_quote_without_the_value() -> None:
    description = "Прибор рассчитан на питание. Напряжение питания 24 В."

    assert not quote_supports("Прибор рассчитан на питание", "24 В", description)


def test_quote_supports_rejects_too_short_quote() -> None:
    assert not quote_supports("24", "24", "Питание 24 В")


@pytest.mark.parametrize(
    "text",
    ["манометр ip65", "манометр g1/4", "манометр m20x1,5", "манометр dm03"],
)
def test_lowercase_invented_code_is_rejected(text: str, dm02_facts: SourceFacts) -> None:
    assert find_unsupported(text, dm02_facts)


@pytest.mark.parametrize(
    "text",
    [
        "класс точности 1,6",
        "Класс точности составляет 1,6",
        "класса точности 100",
        "диаметр корпуса 1,6 мм",
    ],
)
def test_value_of_another_field_is_rejected(text):
    violations = find_unsupported(text, build_source_facts(dm02_product()))
    assert any(v.kind == "binding" for v in violations)


@pytest.mark.parametrize(
    "text",
    [
        "класс точности 1,5",
        "класса точности 1,5 и диаметр корпуса 100 мм",
        "диапазон измерений составляет 0–1,6 МПа",
        "класс точности и диаметр корпуса 100 мм",
        "диаметр корпуса подходит для задачи, 1,6 МПа",
    ],
)
def test_value_next_to_its_own_field_is_allowed(text):
    assert find_unsupported(text, build_source_facts(dm02_product())) == []


@pytest.mark.parametrize("word", ["место 5", "мост 2", "пост 3"])
def test_ordinary_word_is_not_read_as_a_standard(word):
    violations = find_unsupported(word, build_source_facts(dm02_product()))
    assert all(v.kind != "code" for v in violations)


@pytest.mark.parametrize(
    "text",
    [
        "до 1,6 бара",
        "до 1,6 атмосферы",
        "до 1,6 килопаскаля",
        "диаметр корпуса 100 сантиметров",
        "глубина 100 метров",
        "1,5 градуса",
        "погрешность 1,5 процента",
        "до 1.6E1 МПа",
        "1,5‰",
        "корпус Ø150 мм",
        "корпус ø250",
        "степень защиты ІР65",
        "степень защиты ІP65",
        "резьба Ɡ3/4",
        "Класс точности: 1,6",
        "класс точности (1,6)",
        "Класс точности — 1,6",
        "диаметр корпуса: 1,6 МПа",
        "диапазон измерений 0-100 мм",
        "диаметр корпуса сто миллиметров",
        "диаметр корпуса десять сантиметров",
        "класс точности полтора",
        "половина дюйма",
        "в два раза",
        "Производитель Метран",
        "изготовитель Элемер",
        "серия Эконом-Про",
    ],
)
def test_audit_bypass_is_rejected(text, dm02_facts):
    assert find_unsupported(text, dm02_facts), text


@pytest.mark.parametrize(
    "text",
    [
        "диаметр 100 миллиметров",
        "манометр dm 02 100 мм",
        "Манометр DM 02 100 мм G1/2",
        "чтобы лучше оценить применимость",
        "манометр 0 -1,6 МПа",
        "Служба КИП предприятия и инженер АСУ ТП",
        "Манометр показывающий DM 02 Манотомь",
    ],
)
def test_audit_false_alarm_is_gone(text, dm02_facts):
    assert find_unsupported(text, dm02_facts) == [], text


def test_unit_is_not_inherited_across_different_parameters():
    product = ProductInput(
        name="Прибор",
        description="Класс точности 1,5, диаметр корпуса 100 мм. Диапазон 0–1,6 МПа.",
    )
    assert find_unsupported("погрешность 1,5 мм", build_source_facts(product))


def test_commercial_words_are_allowed_only_in_queries(dm02_facts):
    assert find_unsupported("манометр dm 02 цена", dm02_facts, is_query=True) == []
    assert find_unsupported("манометр dm 02 цена", dm02_facts)


def test_value_disputed_by_description_is_not_allowed():
    product = ProductInput(
        name="Преобразователь",
        specs={"диапазон измерений": "0–2,5 МПа"},
        description="По паспорту диапазон измерений 0–25 МПа.",
    )
    facts = build_source_facts(product)
    assert find_unsupported("до 25 МПа", facts)
    assert find_unsupported("диапазон 0–25 МПа", facts)
    assert find_unsupported("диапазон измерений 0–2,5 МПа", facts) == []


def test_conflict_between_specs_and_description_is_found():
    product = ProductInput(
        name="Преобразователь",
        specs={"диапазон измерений": "0–2,5 МПа"},
        description="По паспорту диапазон измерений 0–25 МПа.",
    )
    conflicts = find_conflicts(product)
    assert [(c.field, c.spec_value, c.description_value) for c in conflicts] == [
        ("диапазон измерений", "0–2,5 МПа", "0-25 МПа")
    ]


def test_matching_description_is_not_a_conflict():
    product = ProductInput(
        name="Манометр",
        specs={"класс точности": "1,5"},
        description="Класс точности 1,5 подтверждён паспортом.",
    )
    assert find_conflicts(product) == []


@pytest.mark.parametrize(
    ("quote", "value", "name", "expected"),
    [
        ("Напряжение питания 24 В", "24 В", "питание", True),
        ("Напряжение питания 24 В", "4 В", "питание", False),
        ("Напряжение питания 24 В", "24 мА", "питание", False),
        ("Напряжение питания 24 В", "24 В", "рабочая температура", False),
        ("Напряжение питания от сети", "от сети", "питание", True),
    ],
)
def test_quote_must_back_name_and_whole_value(quote, value, name, expected):
    description = f"Описание прибора. {quote}. Конец."
    assert quote_supports(quote, value, description, name) is expected


@pytest.mark.parametrize(
    "text",
    [
        "технолог нефтегазовой отрасли",
        "для химической промышленности",
        "контроль давления в сосудах под давлением",
        "обеспечивает точное преобразование давления",
        "стабильное электрическое сопротивление",
    ],
)
def test_unproven_application_or_quality_claim_is_rejected(text, dm02_facts):
    assert find_unsupported(text, dm02_facts), text


def test_application_named_in_the_input_is_allowed():
    product = ProductInput(
        name="Манометр", description="Применяется на объектах нефтегазовой отрасли."
    )
    assert find_unsupported("для нефтегазовой отрасли", build_source_facts(product)) == []


@pytest.mark.parametrize(
    "text",
    [
        "погрешность не превышает предела шкалы",
        "соответствует общепринятым габаритам",
        "резьба обеспечивает стандартное подключение",
        "без дополнительных переходников",
        "совместимость с большинством контроллеров",
        "Тип корпуса не указан в технических данных",
        "в источниках данные расходятся",
    ],
)
def test_unproven_property_or_service_note_is_rejected(text, dm02_facts):
    assert find_unsupported(text, dm02_facts), text


def test_word_value_may_differ_by_case_ending():
    quote = "Прибор предназначен для измерения избыточного давления"
    assert quote_supports(quote, "избыточное", quote, "тип давления")
    assert not quote_supports(quote, "абсолютное", quote, "тип давления")


@pytest.mark.parametrize("text", ["через внешнюю резьбу G1/2", "наружная резьба"])
def test_thread_type_absent_from_input_is_rejected(text, dm02_facts):
    assert find_unsupported(text, dm02_facts), text


@pytest.mark.parametrize("text", ["в щите автоматики", "для щитовой установки"])
def test_mounting_place_absent_from_input_is_rejected(text, dm02_facts):
    assert find_unsupported(text, dm02_facts), text


def test_placeholder_from_input_is_rejected_in_buyer_text():
    facts = facts_for(name="Преобразователь", specs={"тип корпуса": "н/д"})
    assert find_unsupported("Тип корпуса н/д уточняется перед заказом", facts)
