"""test_bom.py - checks for normaliser, distances, matcher and mapping."""

from bom_tools import (
    mpn_key,
    levenshtein,
    damerau,
    load_reference,
    match_mpn,
    parse_qty,
    count_refdes,
    validate_mapping,
)
from bom_clean import clean

ROWS, BY_KEY, COLLISIONS = load_reference()


def test_normaliser_has_no_collisions():
    assert COLLISIONS == []


def test_key_ignores_case_space_dash():
    assert mpn_key(" tps5430 dda-r ") == "TPS5430DDAR"


def test_distances_by_hand():
    assert (
        levenshtein("BC847", "BC874") == 2 and damerau("BC847", "BC874") == 1
    )
    assert levenshtein("", "ABC") == 3 and levenshtein("NE555", "NE555") == 0


def test_lookalike_auto_but_fuzzy_never_auto():
    assert match_mpn("STM32F4O7VGT6", ROWS, BY_KEY).status == "auto"
    m = match_mpn("LM2956S-ADJ", ROWS, BY_KEY)
    assert m.status == "review" and m.mpn == "LM2596S-ADJ"


def test_unknown_is_unmatched():
    assert match_mpn("XYZ-123-FOO", ROWS, BY_KEY).status == "unmatched"


def test_qty_and_refdes():
    assert parse_qty("2.500") == 2500 and parse_qty("1k") == 1000
    assert count_refdes("U1-U10") == 10 and count_refdes("U24, U25") == 2


def test_bad_llm_mapping_rejected():
    errs = validate_mapping(
        {"Menge": "qty", "Platz": "qty"}, ["Menge", "Platz"]
    )
    assert errs


def test_whole_bom():
    lines, mapping, _, _ = clean("messy_bom.csv")
    status = [l["status"] for l in lines]
    assert status.count("auto") == 9 and status.count("unmatched") == 1
    assert mapping["Platz"] == "refdes"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
