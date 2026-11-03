from __future__ import annotations

import time
from collections import namedtuple

import pytest

from savia.observability import pii as redact
from savia.observability.pii import (
    CYCLE_MARKER,
    MAX_DEPTH,
    MAX_DEPTH_MARKER,
    REDACTION_FAILED,
    redact_text,
    redact_value,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
EMAIL = "<EMAIL>"
PHONE = "<PHONE>"
IBAN = "<IBAN>"
CREDIT_CARD = "<CARD>"
IPV4 = "<IP>"

SAMPLES_PER_TYPE = 15
REDOS_TIME_BUDGET_S = 2.0


# ---------------------------------------------------------------------------
# GÉNÉRATION DES DONNÉES (déterministe, sans donnée réelle)
# ---------------------------------------------------------------------------
def _group(value: str, size: int = 4, sep: str = " ") -> str:
    return sep.join(value[i : i + size] for i in range(0, len(value), size))


def _iban_with_checksum(country: str, bban: str) -> str:
    """Construit un IBAN valide (clé calculée selon ISO 13616, mod 97)."""
    numeric = "".join(str(int(c, 36)) for c in bban + country + "00")
    check = 98 - int(numeric) % 97
    return f"{country}{check:02d}{bban}"


def _luhn_complete(body: str) -> str:
    """Ajoute le chiffre de contrôle de Luhn à `body`."""
    total = 0
    for i, ch in enumerate(reversed(body)):
        digit = int(ch)
        if i % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return body + str((10 - total % 10) % 10)


EMAILS = [
    "jean.dupont@example.com",
    "marie-claire@exemple.fr",
    "USER@EXAMPLE.COM",
    "j.doe+newsletter@gmail.com",
    "contact@sub.domain.example.org",
    "a@b.co",
    "first_last@company.io",
    "support@mail.example.co.uk",
    "nom.prenom@univ-paris.fr",
    "dev123@startup.ai",
    "x.y.z@example.net",
    "info@my-company.eu",
    "alice@example.museum",
    "bob.martin@orange.fr",
    "sales@example-shop.com",
]

PHONES = [
    "06 12 34 56 78",
    "0612345678",
    "06.12.34.56.78",
    "06-12-34-56-78",
    "+33 6 12 34 56 78",
    "+33612345678",
    "0033 6 12 34 56 78",
    "01 23 45 67 89",
    "+1 (415) 555-2671",
    "+1-415-555-2671",
    "+44 20 7946 0958",
    "+44 7911 123456",
    "+49 151 23456789",
    "+32 470 12 34 56",
    "+34 612 345 678",
]

_IBAN_PARTS = [
    ("FR", "20041010050500013M02606"),
    ("FR", "30006000011234567890189"),
    ("FR", "12345123451234512345123"),
    ("DE", "370400440532013000"),
    ("DE", "100100100123456789"),
    ("ES", "21000418450200051332"),
    ("ES", "00491500051234567892"),
    ("IT", "X0542811101000000123456"),
    ("BE", "539007547034"),
    ("NL", "ABNA0417164300"),
    ("NL", "INGB0001234567"),
    ("GB", "NWBK60161331926819"),
    ("GB", "BARC20040134567890"),
    ("CH", "00762011623852957"),
    ("PT", "000201231234567890154"),
]
# Alternance format compact / format imprimé (groupes de 4)
IBANS = [
    (_group(iban) if i % 2 else iban)
    for i, iban in enumerate(_iban_with_checksum(c, b) for c, b in _IBAN_PARTS)
]

_CARD_BODIES = [
    # Visa (16 chiffres)
    "411111111111111",
    "424242424242424",
    "400000000000000",
    "401288888888188",
    "453201511283036",
    "491688888888888",
    # Mastercard (16 chiffres)
    "555555555555444",
    "510510510510510",
    "545454545454545",
    "535353535353535",
    "525252525252525",
    "543210987654321",
    # Amex (15 chiffres)
    "37144963539843",
    "37828224631000",
    "34111111111111",
]
# Alternance : compact / espaces / tirets
CARDS = []
for _i, _body in enumerate(_CARD_BODIES):
    _pan = _luhn_complete(_body)
    CARDS.append([_pan, _group(_pan), _group(_pan, sep="-")][_i % 3])

IPV4S = [
    "192.168.1.1",
    "10.0.0.1",
    "172.16.254.3",
    "8.8.8.8",
    "203.0.113.42",
    "198.51.100.7",
    "255.255.255.254",
    "1.1.1.1",
    "127.0.0.1",
    "192.0.2.146",
    "85.12.34.56",
    "91.200.1.99",
    "37.59.0.250",
    "62.210.16.8",
    "100.64.0.12",
]

PII_SAMPLES: dict[str, list[str]] = {
    EMAIL: EMAILS,
    PHONE: PHONES,
    IBAN: IBANS,
    CREDIT_CARD: CARDS,
    IPV4: IPV4S,
}

ALL_CASES = [
    pytest.param(placeholder, value, id=f"{placeholder.strip('<>')}-{i:02d}")
    for placeholder, values in PII_SAMPLES.items()
    for i, value in enumerate(values)
]
PER_TYPE = [pytest.param(ph, vals, id=ph.strip("<>")) for ph, vals in PII_SAMPLES.items()]

NO_PII_TEXTS = [
    "Bonjour, comment allez-vous ?",
    "Commande n°12345 expédiée le 12/03/2024.",
    "Version 3.11.2 déployée en production.",
    "Le prix est de 49,99 € TTC.",
    "Réf. produit : ABC-123-XYZ",
    "Rendez-vous à 14h30 en salle B2.",
    "L'arobase (@) est un symbole.",
    "Température : -12.5 degrés",
    "Erreur ligne 42, colonne 7.",
    "Horodatage : 2024-03-12T10:15:30Z",
    "Identifiant : 123e4567-e89b-12d3-a456-426614174000",
]

"""
REDACT_TEXT
"""


# ---------------------------------------------------------------------------
# 0. Garde-fous sur le jeu de données lui-même
# ---------------------------------------------------------------------------
class TestDataset:
    @pytest.mark.parametrize(("placeholder", "values"), PER_TYPE)
    def test_each_type_has_15_distinct_samples(self, placeholder, values):
        assert len(values) == SAMPLES_PER_TYPE
        assert len(set(values)) == SAMPLES_PER_TYPE

    @pytest.mark.parametrize("iban", IBANS)
    def test_generated_ibans_are_valid(self, iban):
        s = iban.replace(" ", "")
        rearranged = s[4:] + s[:4]
        assert int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1

    @pytest.mark.parametrize("card", CARDS)
    def test_generated_cards_pass_luhn(self, card):
        digits = [int(c) for c in card if c.isdigit()][::-1]
        total = sum(
            d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits)
        )
        assert total % 10 == 0


# ---------------------------------------------------------------------------
# 1. Remplacement unitaire : 15 occurrences x 5 types
# ---------------------------------------------------------------------------
class TestSingleOccurrence:
    @pytest.mark.parametrize(("placeholder", "value"), ALL_CASES)
    def test_value_alone_is_replaced(self, placeholder, value):
        assert redact_text(value) == placeholder

    @pytest.mark.parametrize(("placeholder", "value"), ALL_CASES)
    def test_value_in_sentence_keeps_surrounding_text(self, placeholder, value):
        text = f"Bonjour, mon contact est {value}, merci d'avance."
        expected = f"Bonjour, mon contact est {placeholder}, merci d'avance."
        out = redact_text(text)
        assert out == expected
        assert value not in out  # aucune fuite de la donnée brute

    @pytest.mark.parametrize(
        "template",
        ["{v}\n", "\n{v}", "({v})", '"{v}"', "Contact : {v}.", "a\t{v}\tb"],
        ids=["trailing-nl", "leading-nl", "parentheses", "quotes", "final-dot", "tabs"],
    )
    @pytest.mark.parametrize(("placeholder", "value"), ALL_CASES)
    def test_value_next_to_punctuation_and_whitespace(self, placeholder, value, template):
        assert redact_text(template.format(v=value)) == template.format(v=placeholder)


# ---------------------------------------------------------------------------
# 2. Occurrences multiples
# ---------------------------------------------------------------------------
class TestMultipleOccurrences:
    @pytest.mark.parametrize(("placeholder", "values"), PER_TYPE)
    def test_same_value_repeated_is_replaced_everywhere(self, placeholder, values):
        text = " et ".join([values[0]] * 3)
        assert redact_text(text) == " et ".join([placeholder] * 3)

    @pytest.mark.parametrize(("placeholder", "values"), PER_TYPE)
    def test_all_15_values_of_a_type_in_one_text(self, placeholder, values):
        out = redact_text(" ; ".join(values))
        assert out == " ; ".join([placeholder] * SAMPLES_PER_TYPE)

    @pytest.mark.parametrize("idx", range(SAMPLES_PER_TYPE))
    def test_mixed_types_in_one_text(self, idx):
        placeholders = list(PII_SAMPLES)
        values = [PII_SAMPLES[ph][idx] for ph in placeholders]
        out = redact_text("Données : " + " ; ".join(values))
        assert out == "Données : " + " ; ".join(placeholders)


# ---------------------------------------------------------------------------
# 3. Cas négatifs : pas de faux positifs
# ---------------------------------------------------------------------------
class TestNoFalsePositives:
    @pytest.mark.parametrize("text", NO_PII_TEXTS)
    def test_text_without_pii_is_unchanged(self, text):
        assert redact_text(text) == text

    @pytest.mark.parametrize("placeholder", list(PII_SAMPLES))
    def test_existing_placeholders_are_left_untouched(self, placeholder):
        text = f"Valeur déjà masquée : {placeholder}"
        assert redact_text(text) == text


# ---------------------------------------------------------------------------
# 4. Cas limites
# ---------------------------------------------------------------------------
class TestEdgeCases:
    @pytest.mark.parametrize("text", ["", " ", "   \n\t  "], ids=["empty", "space", "whitespace"])
    def test_empty_and_whitespace_only(self, text):
        assert redact_text(text) == text

    def test_unicode_and_emoji_around_pii_are_preserved(self):
        out = redact_text("Écris à jean.dupont@example.com 😀 ça va — très bien")
        assert out == "Écris à <EMAIL> 😀 ça va — très bien"

    def test_multiline_text(self):
        text = f"ligne 1 : {EMAILS[0]}\nligne 2 : {IPV4S[0]}\nligne 3 : rien"
        assert redact_text(text) == "ligne 1 : <EMAIL>\nligne 2 : <IP>\nligne 3 : rien"

    @pytest.mark.parametrize(("placeholder", "value"), ALL_CASES[::5])
    def test_always_returns_str(self, placeholder, value):
        assert isinstance(redact_text(value), str)


# ---------------------------------------------------------------------------
# 5. Propriétés : idempotence et déterminisme
# ---------------------------------------------------------------------------
class TestProperties:
    @pytest.mark.parametrize(("placeholder", "value"), ALL_CASES)
    def test_idempotent(self, placeholder, value):
        text = f"Contact : {value}."
        once = redact_text(text)
        assert redact_text(once) == once

    @pytest.mark.parametrize(("placeholder", "value"), ALL_CASES[::5])
    def test_deterministic(self, placeholder, value):
        assert redact_text(value) == redact_text(value)


# ---------------------------------------------------------------------------
# 6. Robustesse / sécurité : entrées pathologiques (ReDoS)
# ---------------------------------------------------------------------------
PATHOLOGICAL_INPUTS = {
    "long_letters": "a" * 100_000,
    "many_at_signs": "a@" * 50_000,
    "long_digits": "1" * 100_000,
    "digits_and_spaces": "1 " * 50_000,
    "dots": "." * 100_000,
    "dotted_labels_then_at": "a." * 50_000 + "@",
    "phone_prefix_then_noise": "+33" + " 1" * 50_000,
    "iban_like_noise": "FR76" + " 1234" * 20_000,
    "ipv4_like_noise": "1." * 50_000,
}


class TestRobustness:
    @pytest.mark.parametrize("text", PATHOLOGICAL_INPUTS.values(), ids=PATHOLOGICAL_INPUTS.keys())
    def test_pathological_input_is_processed_in_bounded_time(self, text):
        start = time.perf_counter()
        out = redact_text(text)
        elapsed = time.perf_counter() - start
        assert isinstance(out, str)
        assert elapsed < REDOS_TIME_BUDGET_S, f"trop lent : {elapsed:.2f}s (ReDoS possible ?)"


"""
REDACT
"""

TEST_EMAIL = EMAILS[0]
MASKED = EMAIL


# --- Cas de base -----------------------------------------------------------


def test_string_is_redacted():
    assert redact_value(f"mail: {TEST_EMAIL}") == f"mail: {MASKED}"


@pytest.mark.parametrize("leaf", [42, 3.14, True, None, b"bytes", {1, 2}])
def test_non_string_leaves_unchanged(leaf):
    assert redact_value(leaf) == leaf


def test_unexpected_object_returned_as_is():
    class Weird:
        pass

    w = Weird()
    assert redact_value(w) is w


def test_empty_containers():
    assert redact_value({}) == {}
    assert redact_value([]) == []
    assert redact_value(()) == ()


# --- Récursivité et préservation des types ---------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"a": TEST_EMAIL}, {"a": MASKED}),
        ([TEST_EMAIL, 1], [MASKED, 1]),
        ((TEST_EMAIL, 1), (MASKED, 1)),
    ],
)
def test_container_type_preserved(data, expected):
    result = redact_value(data)
    assert result == expected
    assert type(result) is type(data)


def test_mixed_nesting_all_directions():
    data = {
        "u": [TEST_EMAIL, (TEST_EMAIL, {"k": [1, None, TEST_EMAIL]})],
        "t": ([{"x": TEST_EMAIL}],),
    }
    result = redact_value(data)
    assert result == {
        "u": [MASKED, (MASKED, {"k": [1, None, MASKED]})],
        "t": ([{"x": MASKED}],),
    }
    assert isinstance(result["u"], list)
    assert isinstance(result["u"][1], tuple)
    assert isinstance(result["t"], tuple)
    assert isinstance(result["t"][0], list)


def test_namedtuple_preserved():
    Point = namedtuple("Point", "mail n")
    result = redact_value(Point(EMAIL, 3))
    assert isinstance(result, Point)
    assert result == Point(MASKED, 3)


def test_dict_keys_are_not_redacted():
    assert redact_value({TEST_EMAIL: TEST_EMAIL}) == {TEST_EMAIL: MASKED}


# --- Immutabilité de l'entrée ----------------------------------------------


def test_input_is_not_mutated_and_new_object_returned():
    data = {"a": [TEST_EMAIL, (TEST_EMAIL,)]}
    snapshot = repr(data)
    result = redact_value(data)
    assert repr(data) == snapshot
    assert result is not data
    assert result["a"] is not data["a"]


# --- Garde-fous : cycles et profondeur -------------------------------------


def test_list_cycle_is_handled():
    cyc = []
    cyc.append(cyc)
    assert redact_value(cyc) == [CYCLE_MARKER]


def test_dict_cycle_is_handled():
    d = {}
    d["self"] = d
    assert redact_value(d) == {"self": CYCLE_MARKER}


def test_shared_reference_is_not_a_cycle():
    shared = [TEST_EMAIL]
    result = redact_value({"a": shared, "b": shared})
    assert result == {"a": [MASKED], "b": [MASKED]}


def test_depth_limit_returns_marker():
    value = TEST_EMAIL
    for _ in range(MAX_DEPTH + 100):
        value = [value]
    result = redact_value(value)
    for _ in range(MAX_DEPTH):
        result = result[0]
    assert result == MAX_DEPTH_MARKER


def test_depth_just_under_limit_is_fully_redacted():
    value = TEST_EMAIL
    for _ in range(MAX_DEPTH - 1):
        value = [value]
    result = redact_value(value)
    for _ in range(MAX_DEPTH - 1):
        result = result[0]
    assert result == MASKED


# --- Robustesse : ne jamais lever, ne jamais fuiter ------------------------


def test_redact_text_failure_is_fail_closed(monkeypatch):
    def boom(_text):
        raise ValueError("boom")

    monkeypatch.setattr(redact, "redact_text", boom)
    # L'élément en échec est remplacé, le reste de la structure est conservé.
    assert redact_value([TEST_EMAIL, 3]) == [REDACTION_FAILED, 3]


def test_unexpected_structure_error_returns_marker():
    class BadTuple(tuple):
        _fields = ("a",)

        def __new__(cls, *_args):
            raise RuntimeError("cannot rebuild")

    # Un namedtuple "cassé" ne doit pas faire planter, ni renvoyer l'original.
    assert redact_value(tuple.__new__(BadTuple, (TEST_EMAIL,))) == REDACTION_FAILED
