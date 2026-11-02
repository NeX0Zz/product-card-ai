"""Anti-hallucination check: every fact in generated text must exist in the input.

A prompt only asks the model not to invent values. This module enforces it: it extracts
numbers with units, technical codes and risky claims from the generated text and compares
them with the input. The check is strict on purpose, because a false alarm costs one repair
attempt and a missed invented value ends up in a published B2B product card.
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .models import ProductInput

_DASHES = "‐‑‒–—―−"
_LOOKALIKES = str.maketrans("авекмнорстухі", "abekmhopctyxi")
_ALLOWED_TERMS = {"b2b", "b2c", "seo"}
_ALLOWED_NAMES = {"кип", "кипиа", "асу", "тп", "асутп"}
_MAX_TOKEN_CHARS = 60

_UNIT_SHORT = (
    r"мм\s?рт\.?\s?ст\.?|мм\s?вод\.?\s?ст\.?|кгс/см2|кгс/м2|кгс|мпа|кпа|гпа|па|mpa|kpa|hpa|pa|"
    r"мбар|mbar|бар|bar|атм|psi|мкм|мм|см|дм|км|mm|cm|km|°[cсf]?|мкв|мв|кв|мка|ма|ua|мвт|квт|вт|"
    r"kw|mw|mv|kv|ma|гц|кгц|мгц|hz|khz|mhz|ком|мом|ом|кг|мг|kg|мс|ms|сек|мин|час(?:а|ов)?|"
    r"сут(?:ок|ки)?|лет|год(?:а|у)?|мес(?:яц(?:а|ев)?)?|л/мин|л/ч|м3/ч|м3/с|дюйм(?:а|ов)?|"
    r"кбит/с|мбит/с|бит/с|бод|ppm|об/мин|шт"
)
# Units spelled out. "1,6 бара" is the same claim as "1,6 бар" and must not slip past the check.
_UNIT_WORDS: tuple[tuple[str, str], ...] = (
    (r"мегапаскал\w*", "мпа"),
    (r"килопаскал\w*", "кпа"),
    (r"гектопаскал\w*", "гпа"),
    (r"паскал\w*", "па"),
    (r"миллибар\w*", "мбар"),
    (r"бар(?:а|ы|ов|ах|ам)", "бар"),
    (r"атмосфер\w*", "атм"),
    (r"миллиметр\w*", "мм"),
    (r"сантиметр\w*", "см"),
    (r"дециметр\w*", "дм"),
    (r"километр\w*", "км"),
    (r"метр(?:а|ы|ов|ах|ам)?", "м"),
    (r"микрон\w*|микрометр\w*", "мкм"),
    (r"градус\w*", "°c"),
    (r"процент\w*", "%"),
    (r"милливольт\w*", "мв"),
    (r"киловольт\w*", "кв"),
    (r"вольт\w*", "в"),
    (r"миллиампер\w*", "ма"),
    (r"микроампер\w*", "мка"),
    (r"ампер\w*", "а"),
    (r"киловатт\w*", "квт"),
    (r"милливатт\w*", "мвт"),
    (r"ватт\w*", "вт"),
    (r"килогерц\w*", "кгц"),
    (r"мегагерц\w*", "мгц"),
    (r"герц\w*", "гц"),
    (r"килоом\w*", "ком"),
    (r"мегаом\w*", "мом"),
    (r"ом(?:а|ов)", "ом"),
    (r"килограмм\w*", "кг"),
    (r"миллиграмм\w*", "мг"),
    (r"грамм\w*", "г"),
    (r"миллисекунд\w*", "мс"),
    (r"секунд\w*", "с"),
    (r"минут\w*", "мин"),
    (r"литр\w*", "л"),
)
_UNIT_WORD_ALTERNATIVES = "|".join(pattern for pattern, _ in _UNIT_WORDS)
_UNIT = re.compile(
    rf"(?:(?P<word>(?i:{_UNIT_WORD_ALTERNATIVES}))(?![а-яёa-z0-9])"
    rf"|(?P<multi>(?i:{_UNIT_SHORT}))(?![а-яёa-z0-9])"
    r"|(?P<pct>[%‰])"
    r"|(?P<upper>[ВАVAWKК])(?![а-яёА-ЯЁa-zA-Z0-9])"
    r"|(?P<lower>[мгсчлmsh])(?![а-яёА-ЯЁa-zA-Z0-9])(?!\s*[а-яёa-z]))"
)
_UNIT_ALIASES = {
    "mpa": "мпа", "kpa": "кпа", "hpa": "гпа", "pa": "па", "mbar": "мбар", "bar": "бар",
    "mm": "мм", "cm": "см", "km": "км", "m": "м", "°": "°c", "°с": "°c", "mv": "мв", "kv": "кв",
    "ma": "ма", "ua": "мка", "kw": "квт", "mw": "мвт", "w": "вт", "v": "в", "a": "а", "k": "к",
    "hz": "гц", "khz": "кгц", "mhz": "мгц", "kg": "кг", "ms": "мс", "s": "с", "h": "ч",
    "сек": "с", "часа": "час", "часов": "час", "сутки": "сут", "суток": "сут",
    "год": "лет", "года": "лет", "году": "лет", "месяц": "мес", "месяца": "мес",
    "месяцев": "мес", "дюйма": "дюйм", "дюймов": "дюйм",
}  # fmt: skip

_NUMBER = re.compile(
    r"(?<![\w.,/])(?<!\d\s)[-+]?(?:[1-9]\d{0,2}(?: \d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
)
_RANGE_LINK = re.compile(r"\s*(?:-|до)\s*")
# Separators of a list or a range that shares one unit: "0,6; 1,0; 1,6 МПа", "0-1,6 МПа".
_LIST_LINK = re.compile(r"\s*(?:[-;,/]|до|и|или)\s*")
_FRACTION = re.compile(r"(?<![\w.,/])\d+/\d+(?![\w/])")
_DIGIT_GROUP = re.compile(r"\d+(?:[.,]\d+)?")

_CODE_PATTERNS = (
    # Standards: ГОСТ 2405-88, ТУ 4212-001, ISO 9001.
    r"(?<!\w)(?:ГОСТ|ОСТ|СТО|ТУ|ТР ТС|ISO|IEC|МЭК|DIN|EN|ASME|ANSI|ASTM)"
    r"\s*(?:Р\s*)?\d[\d.\-/: ]*\d",
    r"(?<!\w)IP\s?\d{2}[A-Z]?(?!\w)",
    # Threads: G1/2, М20х1,5, NPT 1/2, 1/2 NPT.
    r"(?<![\w/])\d+/\d+\s?(?:NPT|BSP)",
    r"(?<!\w)(?:NPT|BSP)\s?\d+(?:/\d+)?",
    r"(?<!\w)[GRМM]\s?\d+(?:/\d+)?(?:\s?[xXхХ×]\s?\d+(?:[.,]\d+)?)?(?!\w)",
    # Any letter glued to digits: RS-485, DM02, Ø150, ІР65. A hyphen continues the token only
    # before a digit, so "B2B-закупок" stays "B2B".
    r"(?<!\w)[^\W\d_]\w*-?\d[\w/]*(?:-\d[\w/]*)*",
)
# Case-insensitive: search queries are lowercase, and "ip65" is the same claim as "IP65".
_CODE = re.compile("|".join(f"(?:{pattern})" for pattern in _CODE_PATTERNS), re.I)
_LATIN_WORD = re.compile(r"(?<![\w-])[A-Za-z][A-Za-z+#-]*[A-Za-z+#](?![\w])")
_CAPITALIZED_WORD = re.compile(r"(?<![\w-])[А-ЯЁ][а-яёА-ЯЁ-]+")
_SENTENCE_START = re.compile(r"(?:^|[.!?:\n]\s*|-\s+)$")
_CONTACT = re.compile(
    r"https?://|www\.|[\w.+-]+@[\w-]+\.\w+|\b[\w-]+\.(?:ru|com|net|org|рф)\b|\+?\d[\d\s()-]{9,}\d",
    re.I,
)

_COMMERCIAL_LABEL = "коммерческие условия"
# A placeholder like "н/д" may be in the input, but it must never reach the buyer's text.
_SERVICE_NOTE_LABEL = "служебная пометка в тексте для покупателя"
# Technical and commercial claims that carry no digits, so number checks cannot see them.
# Each is allowed only when the same stem occurs in the input.
_CLAIMS: tuple[tuple[str, str], ...] = (
    (r"взрыв\w*", "взрывозащита"),
    (r"нержаве\w*|латун\w*|бронз\w*|алюмини\w*|титан\w*|сталь\w*|стальн\w*|медн\w*", "материал"),
    (r"пластик\w*|полимер\w*|поликарбонат\w*|полиамид\w*|керамик\w*", "материал"),
    (r"глицерин\w*|гидрозаполн\w*|маслозаполн\w*", "заполнение"),
    (r"поверк\w*|поверочн\w*|межповерочн\w*|госреестр\w*|реестр\w*|калибров\w*", "поверка"),
    (r"сертифи\w*|декларац\w*|гост\w*|\bту\b|\biso\b", "соответствие стандартам"),
    (r"гаранти\w*", "гарантия"),
    (r"(?:внешн|наружн|внутренн)\w+\s+резьб\w*", "тип резьбы"),
    (r"вибро\w*|ударопроч\w*|пылезащ\w*|влагозащ\w*|пылевлаго\w*|герметичн\w*", "защита"),
    (
        r"морозост\w*|термост\w*|жаропроч\w*|коррозионн\w*|антикоррозийн\w*|агрессивн\w*",
        "стойкость",
    ),
    (r"беспроводн\w*|wi-?fi|bluetooth|modbus|hart|profibus|ethernet", "интерфейс"),
    (r"высокоточн\w*|высок\w+ точност\w*|прецизионн\w*", "оценка точности"),
    (
        r"лучш(?:ий|ая|ее|ие|его|ей|ему|ую|их|им|ими)\b|сам\w+ точн\w*|лидер\w*|уникальн\w*"
        r"|непревзойд\w*|идеальн\w*|№\s?1",
        "реклама",
    ),
    (r"надежн\w*|долговечн\w*|безотказн\w*|долгий срок\w*|стабильн\w*", "оценка надёжности"),
    (
        r"точн(?:ое|ого|ому|ая|ой|ую|ый|ым|ые|ых|ыми)\s+"
        r"(?:измерен|преобразован|отображен|показан|контрол|определен|значен)\w*",
        "оценка точности",
    ),
    # Unproven benefits: the words models use to turn a plain value into a product property.
    (
        r"погрешност\w*|стандартн\w*|общепринят\w*|совместим\w*|большинств\w*|достаточн\w*"
        r"|типов\w*|универсальн\w*|обеспеч\w*|облегча\w*|упроща\w*|гарантиру\w*|благодаря"
        r"|оптимальн\w*|ограниченн\w*|панел\w*|стойк[аеиу]\b|щит\w*"
        r"|без дополнительн\w*|без переходник\w*|хорош\w+ читаем\w*|удобн\w*",
        "неподтверждённое свойство",
    ),
    (
        r"не указан\w*|не приведен\w*|расхожден\w*|расходятся|отсутству\w*|требует уточнения"
        r"|необходимо уточнить|нужно уточнить|в источниках|разные значения|\bн/д\b|уточня\w*",
        _SERVICE_NOTE_LABEL,
    ),
    (
        r"нефт\w*|газов\w*|химическ\w*|энергетик\w*|теплоэнерг\w*|пищев\w*|фармацевт\w*"
        r"|металлург\w*|машиностро\w*|судостро\w*|водоснабж\w*|водоканал\w*|отоплен\w*"
        r"|водян\w*|паров\w*|бытов\w*|пневм\w*|гидравл\w*"
        r"|сосуд\w* под давлением|\bжкх\b|\bтэц\b|\bтэк\b",
        "отрасль или область применения",
    ),
    (r"в наличии|доставк\w*|скидк\w*|\bцен[аыуе]\b|\bцены\b|стоимост\w*", _COMMERCIAL_LABEL),
    (
        r"\b(?:два|две|двух|три|трех|четыре|четырех|пять|пяти|шесть|шести|семь|семи|восемь|восьми"
        r"|девять|девяти|десять|десяти|сто|ста|сотн\w*|сотен|тысяч\w*|полтора|полторы|половин\w*)\b",
        "количество словами",
    ),
)
_CLAIM_PATTERNS = tuple((re.compile(rf"(?<!\w)(?:{p})", re.I), label) for p, label in _CLAIMS)


def _normalize(text: str) -> str:
    """Same text form for input and output: width, dashes, spaces and letter ё."""
    text = unicodedata.normalize("NFKC", text).replace("⁄", "/").replace(" ", " ")
    for dash in _DASHES:
        text = text.replace(dash, "-")
    # NFKC turns an ellipsis into dots, which would hide the upper bound of "0…150 °C".
    text = re.sub(r"\.{2,}", " - ", text)
    return text.replace("ё", "е").replace("Ё", "Е")


def _canon(text: str) -> str:
    """Compact form for code matching: RS-485 equals rs485, Cyrillic С equals Latin C."""
    lowered = _normalize(text).casefold().translate(_LOOKALIKES)
    return re.sub(r"[^0-9a-zа-я]", "", lowered)


def _number_key(raw: str) -> str:
    compact = raw.replace(" ", "").replace(",", ".")
    try:
        number = Decimal(compact)
    except InvalidOperation:
        return compact
    return format(number.normalize(), "f") if number else "0"


def _unit_key(match: re.Match[str]) -> str:
    word = match.group("word")
    if word:
        return next(canon for pattern, canon in _UNIT_WORDS if re.fullmatch(pattern, word, re.I))
    key = re.sub(r"\s", "", match.group(0).casefold())
    return _UNIT_ALIASES.get(key, key)


def _match_unit(text: str, end: int) -> re.Match[str] | None:
    position = end
    while position < len(text) and text[position] == " ":
        position += 1
    return _UNIT.match(text, position)


@dataclass(frozen=True)
class FoundNumber:
    raw: str
    key: str
    unit: str | None


def _find_numbers(text: str) -> list[FoundNumber]:
    found = []
    for match in _NUMBER.finditer(text):
        unit = _match_unit(text, match.end())
        raw = text[match.start() : unit.end()] if unit else match.group(0)
        found.append(
            FoundNumber(raw, _number_key(match.group(0)), _unit_key(unit) if unit else None)
        )
    return found


def _has_stem(word: str, text: str) -> bool:
    """Check for a word by its stem from the word start: "Метран" must not match "диаметр"."""
    # A phrase needs an exact match: "класс точности" doesn't prove "точное измерение".
    if " " in word:
        return word.casefold() in text
    stem = word.casefold()[:5]
    ending = "" if len(stem) == 5 else r"(?!\w)"
    return re.search(rf"(?<!\w){re.escape(stem)}{ending}", text) is not None


def _strip_spans(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(" ", text)


def _pairs_with_shared_units(text: str) -> set[tuple[str, str]]:
    """Pair numbers with units. A bare number inherits a unit only inside a list or a range."""
    numbers = list(_NUMBER.finditer(text))
    pairs: set[tuple[str, str]] = set()
    inherited: str | None = None
    for index in range(len(numbers) - 1, -1, -1):
        match = numbers[index]
        unit_match = _match_unit(text, match.end())
        if unit_match:
            inherited = _unit_key(unit_match)
        elif index + 1 < len(numbers):
            between = text[match.end() : numbers[index + 1].start()]
            if not _LIST_LINK.fullmatch(between):
                inherited = None
        else:
            inherited = None
        if inherited:
            pairs.add((_number_key(match.group(0)), inherited))
    return pairs


def _key_pattern(key: str) -> re.Pattern[str] | None:
    """Match a spec name in any grammatical case: "класс точности", "класса точности"."""
    words = re.findall(r"[а-яa-z]+", _normalize(key).casefold())
    if not words:
        return None
    stems = (word[: max(4, len(word) - 2)] for word in words)
    return re.compile(r"\s+".join(rf"{stem}\w*" for stem in stems), re.I)


@dataclass(frozen=True)
class SpecBinding:
    """Numbers allowed right after a spec name, so a value can't migrate between fields."""

    key: str
    value: str
    pattern: re.Pattern[str]
    numbers: frozenset[str]


@dataclass(frozen=True)
class SourceFacts:
    numbers: frozenset[str]
    fractions: frozenset[str]
    pairs: frozenset[tuple[str, str]]
    canon: str
    lowered: str
    bindings: tuple[SpecBinding, ...] = ()


def _build_bindings(product: ProductInput) -> tuple[SpecBinding, ...]:
    bindings = []
    for key, value in product.specs.items():
        pattern = _key_pattern(key)
        numbers = _find_numbers(_strip_spans(_normalize(value), _CODE))
        if pattern and numbers:
            allowed = frozenset(number.key for number in numbers)
            bindings.append(SpecBinding(key, value, pattern, allowed))
    return tuple(bindings)


def _numbers_after(text: str, start: int) -> tuple[list[tuple[str, str]], int]:
    """Read a number or a whole range at `start`.

    Returns (raw token, number key) pairs and the position where the range ends.
    """
    found = []
    position = end = start
    while (match := _NUMBER.match(text, position)) is not None:
        found.append((match.group(0), _number_key(match.group(0))))
        unit = _match_unit(text, match.end())
        end = unit.end() if unit else match.end()
        link = _RANGE_LINK.match(text, end)
        if not link:
            break
        position = link.end()
    return found, end


@dataclass(frozen=True)
class MisboundNumber:
    binding: SpecBinding
    field_text: str
    raw: str
    key: str
    fragment: str


def _find_misbound_numbers(text: str, bindings: tuple[SpecBinding, ...]) -> list[MisboundNumber]:
    """Find "класс точности 1,6" when the input gives this field a different value."""
    found = []
    for binding in bindings:
        for match in binding.pattern.finditer(text):
            # A short gap without clause breaks: "класс точности составляет 1,5", "класс: 1,5".
            gap = re.match(r"[^\d.,;)\n]{0,24}", text[match.end() :])
            start = match.end() + (gap.end() if gap else 0)
            between = text[match.end() : start]
            if any(other.pattern.search(between) for other in bindings):
                continue
            numbers, end = _numbers_after(text, start)
            for raw, key in numbers:
                if key not in binding.numbers and key.lstrip("-") not in binding.numbers:
                    found.append(MisboundNumber(binding, match.group(0), raw, key, text[start:end]))
    return found


@dataclass(frozen=True)
class SpecConflict:
    """A spec value and the description disagree about the same parameter."""

    field: str
    spec_value: str
    description_value: str


def find_conflicts(product: ProductInput) -> list[SpecConflict]:
    description = _strip_spans(_normalize(product.description), _CODE)
    misbound = _find_misbound_numbers(description, _build_bindings(product))
    # One range can hold several wrong numbers, but it is one conflict.
    unique = {(m.binding.key, m.binding.value, m.fragment) for m in misbound}
    return [SpecConflict(*item) for item in sorted(unique)]


def build_source_facts(product: ProductInput) -> SourceFacts:
    numbers: set[str] = set()
    fractions: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    description = _normalize(product.description)

    def collect(text: str) -> None:
        fractions.update(re.findall(r"\d+/\d+", text))
        numbers.update(_number_key(token) for token in _DIGIT_GROUP.findall(text))
        stripped = _strip_spans(text, _CODE)
        numbers.update(number.key for number in _find_numbers(stripped))
        pairs.update(_pairs_with_shared_units(stripped))

    collect(_normalize(product.name))
    collect(_normalize(product.manufacturer))
    for key, value in product.specs.items():
        clean_key, clean_value = _normalize(key), _normalize(value)
        collect(clean_key)
        collect(clean_value)
        # The unit may be in the name of the spec: "Диаметр, мм": "100".
        unit_in_key = re.search(r"[,(]\s*([^,()]+?)\s*\)?\s*$", clean_key)
        unit_match = _UNIT.fullmatch(unit_in_key.group(1)) if unit_in_key else None
        if unit_match:
            for number in _find_numbers(_strip_spans(clean_value, _CODE)):
                pairs.add((number.key, _unit_key(unit_match)))

    spec_numbers = frozenset(numbers)
    collect(description)

    # A value the description disputes must not be used until a person resolves the conflict.
    bindings = _build_bindings(product)
    misbound = _find_misbound_numbers(_strip_spans(description, _CODE), bindings)
    disputed = {item.key for item in misbound} - spec_numbers
    numbers -= disputed
    pairs = {pair for pair in pairs if pair[0] not in disputed}

    full_text = " ".join(
        [_normalize(product.name), _normalize(product.manufacturer), description]
        + [_normalize(f"{key}: {value}") for key, value in product.specs.items()]
    )
    return SourceFacts(
        numbers=frozenset(numbers),
        fractions=frozenset(fractions),
        pairs=frozenset(pairs),
        canon=_canon(full_text),
        lowered=full_text.casefold(),
        bindings=bindings,
    )


@dataclass(frozen=True)
class Violation:
    kind: str
    token: str
    message: str


def find_unsupported(text: str, facts: SourceFacts, *, is_query: bool = False) -> list[Violation]:
    """Return every fact in `text` that the input does not back up.

    A search query may hold words like "цена": there they describe user intent, not a claim.
    """
    normalized = _normalize(text)
    violations: list[Violation] = []

    def add(kind: str, token: str, message: str) -> None:
        token = token.strip()[:_MAX_TOKEN_CHARS]
        if not any(v.kind == kind and v.token == token for v in violations):
            violations.append(Violation(kind, token, message))

    for match in _CONTACT.finditer(normalized):
        add("contact", match.group(0), f"«{match.group(0)}»: ссылки и контакты в тексте запрещены.")
    without_contacts = _strip_spans(normalized, _CONTACT)

    for match in _CODE.finditer(without_contacts):
        token = match.group(0)
        if _canon(token) not in facts.canon and _canon(token) not in _ALLOWED_TERMS:
            add("code", token, f"Обозначения «{token}» нет во входных данных: удалите его.")
    without_codes = _strip_spans(without_contacts, _CODE)

    for match in _FRACTION.finditer(without_codes):
        if match.group(0) not in facts.fractions:
            add("number", match.group(0), f"Значения «{match.group(0)}» нет во входных данных.")
    without_fractions = _strip_spans(without_codes, _FRACTION)

    for number in _find_numbers(without_fractions):
        if number.key not in facts.numbers:
            add(
                "number", number.raw, f"Значения «{number.raw}» нет во входных данных: удалите его."
            )
        elif number.unit and (number.key, number.unit) not in facts.pairs:
            add(
                "unit",
                number.raw,
                f"Значение «{number.raw}» с такой единицей не указано во входных данных: "
                "не пересчитывайте и не добавляйте единицы.",
            )

    for item in _find_misbound_numbers(without_fractions, facts.bindings):
        add(
            "binding",
            f"{item.field_text} {item.raw}",
            f"«{item.field_text} {item.raw}»: во входных данных у этого параметра другое значение.",
        )

    for match in _LATIN_WORD.finditer(without_fractions):
        token = match.group(0)
        canon = _canon(token)
        if canon not in _ALLOWED_TERMS and canon not in facts.canon:
            add("latin", token, f"Термина «{token}» нет во входных данных: удалите его.")

    for match in _CAPITALIZED_WORD.finditer(without_fractions):
        token = match.group(0)
        lowered = token.casefold()
        starts_sentence = _SENTENCE_START.search(without_fractions[: match.start()])
        if starts_sentence or lowered in _ALLOWED_NAMES or _has_stem(token, facts.lowered):
            continue
        add("name", token, f"Названия «{token}» нет во входных данных: удалите его.")

    for pattern, label in _CLAIM_PATTERNS:
        if is_query and label == _COMMERCIAL_LABEL:
            continue
        for match in pattern.finditer(normalized):
            if label == _SERVICE_NOTE_LABEL or not _has_stem(match.group(0), facts.lowered):
                add(
                    "claim",
                    match.group(0),
                    f"Утверждение «{match.group(0)}» ({label}) "
                    "не подтверждено входными данными: удалите его.",
                )
    return violations


def quote_supports(quote: str, value: str, description: str, name: str = "") -> bool:
    """Check that a quote is really in the description and backs the extracted name and value."""
    if len(_canon(quote)) < 3 or _canon(quote) not in _canon(description):
        return False

    clean_quote = _normalize(quote).casefold()
    clean_value = _normalize(value).casefold().strip()
    if name:
        stems = [word[:4] for word in re.findall(r"[а-яa-z]{4,}", _normalize(name).casefold())]
        if stems and not any(stem in clean_quote for stem in stems):
            return False

    value_numbers = _find_numbers(_strip_spans(clean_value, _CODE))
    if not value_numbers:
        # Words may differ by case ending: "избыточное" is backed by "избыточного давления".
        words = re.findall(r"[а-яa-z]+", clean_value)
        stems = [word[: max(4, len(word) - 2)] for word in words]
        return bool(stems) and all(stem in clean_quote for stem in stems)

    stripped_quote = _strip_spans(clean_quote, _CODE)
    quote_keys = {number.key for number in _find_numbers(stripped_quote)}
    quote_pairs = _pairs_with_shared_units(stripped_quote)
    return all(
        number.key in quote_keys and (not number.unit or (number.key, number.unit) in quote_pairs)
        for number in value_numbers
    )
