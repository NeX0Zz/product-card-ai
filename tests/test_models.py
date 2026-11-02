import json

import pytest

from product_card.models import (
    KNOWN_FIELDS,
    InputError,
    clean_text,
    load_json,
    parse_product,
)


def parse(raw: object) -> tuple[dict[str, object], list[str]]:
    product, ignored = parse_product(raw)
    return product.model_dump(), ignored


def error_fields(raw: object) -> list[str]:
    with pytest.raises(InputError) as caught:
        parse_product(raw)
    return [item.field for item in caught.value.details]


def test_valid_object_is_parsed() -> None:
    product, ignored = parse(
        {
            "name": "Манометр",
            "manufacturer": "Манотомь",
            "specs": {"класс": "1,5"},
            "description": "d",
        }
    )

    assert product == {
        "name": "Манометр",
        "manufacturer": "Манотомь",
        "specs": {"класс": "1,5"},
        "description": "d",
    }
    assert ignored == []


def test_null_and_missing_fields_become_empty() -> None:
    product, _ = parse(
        {"name": "Манометр", "manufacturer": None, "specs": None, "description": None}
    )

    assert product["manufacturer"] == ""
    assert product["specs"] == {}
    assert product["description"] == ""


def test_numbers_and_booleans_are_coerced_to_text() -> None:
    product, _ = parse(
        {
            "name": "X",
            "specs": {"целое": 5, "дробное": 1.5, "да": True, "нет": False, "пусто": None},
        }
    )

    assert product["specs"] == {
        "целое": "5",
        "дробное": "1.5",
        "да": "да",
        "нет": "нет",
        "пусто": "",
    }


def test_number_as_name_is_coerced() -> None:
    product, _ = parse({"name": 12345})

    assert product["name"] == "12345"


@pytest.mark.parametrize("nested", [["a"], {"a": "b"}, [[]], {}])
@pytest.mark.parametrize("field", ["name", "manufacturer", "description"])
def test_nested_value_in_text_field_is_rejected(field: str, nested: object) -> None:
    assert error_fields({"name": "X", field: nested}) == [field]


@pytest.mark.parametrize("nested", [["a"], {"a": "b"}])
def test_nested_value_in_spec_is_rejected(nested: object) -> None:
    assert error_fields({"name": "X", "specs": {"диапазон": nested}}) == ["specs.диапазон"]


@pytest.mark.parametrize("specs", [["a", "b"], "строка", 5])
def test_non_object_specs_are_rejected(specs: object) -> None:
    assert error_fields({"name": "X", "specs": specs}) == ["specs"]


@pytest.mark.parametrize("root", [[], [{"name": "X"}], "text", 5, None, True])
def test_non_object_root_is_rejected(root: object) -> None:
    with pytest.raises(InputError) as caught:
        parse_product(root)

    assert caught.value.code == "invalid_product"


@pytest.mark.parametrize(
    ("raw", "field"),
    [
        ({"name": "а" * 301}, "name"),
        ({"manufacturer": "а" * 201}, "manufacturer"),
        ({"description": "а" * 20_001}, "description"),
        ({"specs": {f"k{i}": "v" for i in range(101)}}, "specs"),
        ({"specs": {"к" * 201: "v"}}, "specs.{}".format("к" * 60)),
        ({"specs": {"k": "в" * 1001}}, "specs.k"),
    ],
)
def test_values_over_limits_are_rejected(raw: dict[str, object], field: str) -> None:
    assert error_fields(raw) == [field]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"name": "а" * 300}, "name"),
        ({"manufacturer": "а" * 200}, "manufacturer"),
        ({"description": "а" * 20_000}, "description"),
    ],
)
def test_values_at_the_limit_are_accepted(raw: dict[str, object], expected: str) -> None:
    product, _ = parse(raw)

    assert len(str(product[expected])) > 0


def test_limits_of_specs_count_key_and_value_are_inclusive() -> None:
    product, _ = parse({"specs": {**{f"k{i}": "v" for i in range(99)}, "к" * 200: "в" * 1000}})

    assert len(product["specs"]) == 100  # type: ignore[arg-type]


def test_all_errors_are_collected_at_once() -> None:
    fields = error_fields({"name": ["x"], "manufacturer": {"a": 1}, "specs": []})

    assert fields == ["name", "manufacturer", "specs"]


def test_empty_spec_key_is_rejected() -> None:
    assert error_fields({"name": "X", "specs": {"  ": "v"}}) == ["specs"]


def test_control_and_zero_width_characters_are_removed() -> None:
    raw = {
        "name": "Ма​но\u0000метр‮",
        "manufacturer": "﻿Мано­томь",
        "specs": {"диа​пазон": "0\x07–1⁠,6 МПа"},
        "description": "Строка\x1b один",
    }

    product, _ = parse(raw)

    assert product == {
        "name": "Манометр",
        "manufacturer": "Манотомь",
        "specs": {"диапазон": "0–1,6 МПа"},
        "description": "Строка один",
    }


def test_newlines_and_tabs_survive_cleaning() -> None:
    assert clean_text("a\nb\tc") == "a\nb\tc"


def test_text_is_trimmed() -> None:
    product, _ = parse({"name": "  Манометр \n"})

    assert product["name"] == "Манометр"


def test_unknown_fields_are_reported_sorted() -> None:
    _, ignored = parse({"name": "X", "price": 100, "color": "red", "sku": "1"})

    assert ignored == ["color", "price", "sku"]


def test_unknown_field_names_are_cleaned_and_truncated() -> None:
    _, ignored = parse({"name": "X", "по​ле" + "я" * 100: 1})

    assert ignored == ["поле" + "я" * 56]


def test_known_fields_are_not_reported_as_unknown() -> None:
    raw = {field: {} if field == "specs" else "" for field in KNOWN_FIELDS}

    assert parse(raw)[1] == []


def test_invalid_json_reports_line_and_column() -> None:
    with pytest.raises(InputError) as caught:
        load_json('{"name": "X",\n  "specs" oops}')

    assert caught.value.code == "invalid_json"
    assert "строка 2" in caught.value.message
    assert "позиция 11" in caught.value.message


def test_invalid_json_message_does_not_echo_content() -> None:
    with pytest.raises(InputError) as caught:
        load_json('{"name": "TOPSECRET-VALUE",\n "specs": }')

    assert "TOPSECRET" not in caught.value.message
    assert "TOPSECRET" not in str(caught.value)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_input_is_invalid_json(text: str) -> None:
    with pytest.raises(InputError) as caught:
        load_json(text)

    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_constants_that_are_not_numbers_are_rejected(constant: str) -> None:
    with pytest.raises(InputError) as caught:
        load_json(f'{{"name": "X", "specs": {{"p": {constant}}}}}')

    assert caught.value.code == "invalid_json"
    assert constant not in caught.value.message


def test_overflowing_number_is_not_turned_into_inf() -> None:
    raw = load_json('{"name": "X", "specs": {"давление": 1e400}}')

    with pytest.raises(InputError):
        parse_product(raw)


def test_huge_integer_does_not_crash() -> None:
    with pytest.raises(InputError) as caught:
        load_json('{"name": ' + "9" * 10_000 + "}")

    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize("opening", ["[", '{"a":'])
def test_deeply_nested_json_does_not_crash_the_decoder(opening: str) -> None:
    with pytest.raises(InputError) as caught:
        load_json(opening * 200_000)

    assert caught.value.code == "invalid_json"


def test_deeply_nested_value_in_a_field_is_rejected_without_crash() -> None:
    nested = json.loads("[" * 400 + "]" * 400)

    assert error_fields({"name": nested}) == ["name"]


def test_valid_json_text_round_trip() -> None:
    assert load_json('{"name": "Манометр"}') == {"name": "Манометр"}


def test_keys_equal_after_cleaning_are_reported():
    with pytest.raises(InputError) as error:
        parse_product({"specs": {"давление": "1", "дав​ление": "2"}})
    assert any("повторяется" in item.message for item in error.value.details)
