import pytest

from product_card.completeness import (
    build_characteristics,
    check_completeness,
    expected_specs,
    is_placeholder,
)
from product_card.models import NO_DATA, Characteristic, ProductInput

from .fakes import dm02_product

LONG_DESCRIPTION = (
    "Прибор предназначен для измерения избыточного давления в технологических средах."
)


def labels(product: ProductInput) -> set[str]:
    return {spec.label for spec in expected_specs(product)}


def gap_labels(product: ProductInput) -> set[str]:
    return {spec.label for spec in check_completeness(product).gaps}


def test_mechanical_gauge_does_not_expect_power_or_interfaces() -> None:
    assert labels(dm02_product()) == {
        "диапазон измерений",
        "класс точности",
        "присоединение",
        "тип корпуса",
    }


@pytest.mark.parametrize(
    "name", ["Датчик давления X", "Преобразователь давления", "Манометр цифровой", "Контроллер"]
)
def test_electronic_instrument_expects_power_and_interfaces(name: str) -> None:
    assert {"питание", "интерфейсы"} <= labels(dm02_product(name=name))


def test_complete_mechanical_gauge_has_no_gaps() -> None:
    assert gap_labels(dm02_product()) == set()


def test_electronic_instrument_with_gauge_specs_misses_power_and_interfaces() -> None:
    product = dm02_product(name="Датчик давления X")

    assert gap_labels(product) == {"питание", "интерфейсы"}


@pytest.mark.parametrize(
    "placeholder", ["н/д", "-", "—", "уточняется", "", "  ", "N/A", "Нет данных"]
)
def test_placeholder_values_count_as_empty(placeholder: str) -> None:
    assert is_placeholder(placeholder)
    product = dm02_product(specs={"диапазон измерений": placeholder, "класс точности": "1,5"})
    result = check_completeness(product)

    assert "диапазон измерений" in {gap.label for gap in result.gaps}
    assert "specs.диапазон измерений" in {item.field for item in result.missing}
    assert "диапазон измерений" not in result.filled_specs


@pytest.mark.parametrize("value", ["0", "1,5", "нет", "н/д значение"])
def test_real_values_are_not_placeholders(value: str) -> None:
    assert not is_placeholder(value)


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("Предел измерений", "диапазон измерений"),
        ("Шкала", "диапазон измерений"),
        ("Погрешность", "класс точности"),
        ("Точность", "класс точности"),
        ("Резьба", "присоединение"),
        ("Штуцер", "присоединение"),
        ("Корпус", "тип корпуса"),
    ],
)
def test_aliases_satisfy_expected_specs(key: str, label: str) -> None:
    product = dm02_product(specs={key: "значение"})

    assert label not in gap_labels(product)


@pytest.mark.parametrize(
    ("key", "label"),
    [("Напряжение питания", "питание"), ("Выходной сигнал", "интерфейсы"), ("HART", "интерфейсы")],
)
def test_aliases_satisfy_electronic_expectations(key: str, label: str) -> None:
    product = dm02_product(name="Датчик давления", specs={key: "значение"})

    assert label not in gap_labels(product)


def test_missing_name_blocks_generation() -> None:
    result = check_completeness(dm02_product(name=""))

    assert result.blocking
    assert "name" in {item.field for item in result.missing}


def test_no_facts_blocks_generation() -> None:
    result = check_completeness(dm02_product(specs={"диапазон": "н/д"}, description=""))

    assert result.blocking
    assert "specs, description" in {item.field for item in result.missing}


def test_description_alone_is_enough_to_proceed() -> None:
    result = check_completeness(dm02_product(specs={}))

    assert not result.blocking


def test_specs_alone_are_enough_to_proceed() -> None:
    result = check_completeness(dm02_product(description=""))

    assert not result.blocking
    assert "description" in {item.field for item in result.missing}


def test_missing_manufacturer_is_reported_but_not_blocking() -> None:
    result = check_completeness(dm02_product(manufacturer=""))

    assert not result.blocking
    assert "manufacturer" in {item.field for item in result.missing}


def test_short_description_is_reported() -> None:
    result = check_completeness(dm02_product(description="Манометр."))

    assert "description" in {item.field for item in result.missing}


def test_complete_product_reports_nothing_missing() -> None:
    result = check_completeness(dm02_product(description=LONG_DESCRIPTION))

    assert result.missing == []
    assert result.gaps == []


def test_characteristics_hold_exact_input_values() -> None:
    product = dm02_product()

    items, missing = build_characteristics(product, check_completeness(product), [])

    assert [(c.name, c.value, c.source) for c in items] == [
        ("производитель", "Манотомь", "specs"),
        ("диапазон измерений", "0–1,6 МПа", "specs"),
        ("класс точности", "1,5", "specs"),
        ("присоединение", "G1/2", "specs"),
        ("диаметр корпуса", "100 мм", "specs"),
    ]
    assert missing == []


def test_characteristics_show_no_data_for_placeholder_and_missing_manufacturer() -> None:
    product = dm02_product(
        manufacturer="", specs={"диапазон измерений": "н/д", "класс точности": "1,5"}
    )

    items, _ = build_characteristics(product, check_completeness(product), [])

    by_name = {c.name: c for c in items}
    assert (by_name["производитель"].value, by_name["производитель"].source) == (NO_DATA, "none")
    assert (by_name["диапазон измерений"].value, by_name["диапазон измерений"].source) == (
        NO_DATA,
        "none",
    )


def test_characteristics_add_a_no_data_row_for_each_gap() -> None:
    product = dm02_product(name="Датчик давления")

    items, missing = build_characteristics(product, check_completeness(product), [])

    by_name = {c.name: c for c in items}
    assert by_name["питание"].value == NO_DATA
    assert by_name["интерфейсы"].value == NO_DATA
    assert {m.field for m in missing} == {"питание", "интерфейсы"}


def test_extracted_value_never_overrides_a_spec_with_the_same_name() -> None:
    product = dm02_product()
    extracted = [
        Characteristic(name="Диапазон измерений", value="0–25 МПа", source="description", quote="q")
    ]

    items, _ = build_characteristics(product, check_completeness(product), extracted)

    ranges = [c for c in items if c.name.casefold() == "диапазон измерений"]
    assert [(c.value, c.source) for c in ranges] == [("0–1,6 МПа", "specs")]


def test_extracted_value_fills_a_gap_and_removes_it_from_missing() -> None:
    product = dm02_product(specs={"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5"})
    extracted = [
        Characteristic(name="присоединение", value="G1/2", source="description", quote="q")
    ]

    items, missing = build_characteristics(product, check_completeness(product), extracted)

    assert ("присоединение", "G1/2", "description") in [(c.name, c.value, c.source) for c in items]
    assert "присоединение" not in {m.field for m in missing}
    assert "тип корпуса" in {m.field for m in missing}


def test_empty_expected_spec_gets_one_row():
    product = ProductInput(
        name="Преобразователь давления",
        specs={"тип корпуса": "н/д"},
        description="Описание прибора.",
    )
    rows, missing = build_characteristics(product, check_completeness(product), [])
    assert [row.name for row in rows].count("тип корпуса") == 1
    assert all(item.field != "тип корпуса" for item in missing)


@pytest.mark.parametrize("ending", ["...", "…"])
def test_cut_off_description_is_reported(ending):
    product = ProductInput(
        name="Манометр",
        specs={"класс точности": "1,5"},
        description=f"Прибор предназначен для измерения давления{ending}",
    )
    assert any("обрывается" in item.message for item in check_completeness(product).missing)
