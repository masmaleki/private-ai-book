"""test_extract.py - regression tests for the deterministic parts (chapter 21).
Run: python3 test_extract.py   (or: pytest test_extract.py)
"""
from schemas import RfqExtraction
from extract import grounding_errors, strip_fences
from qty import parse_qty

EMAIL = "Bitte 2.500 Stk. STM32F407VGT6 und 10k TPS5430DDAR anbieten."  # example

def make(mpn, qty):
    return RfqExtraction.model_validate({
        "lines": [{"mpn": mpn, "quantity": qty}],
        "priority": "unknown", "language": "de"})

def test_parse_qty():
    assert parse_qty("2.500") == 2500 and parse_qty("10k") == 10000
    assert parse_qty("2,500") == 2500 and parse_qty("400") == 400

def test_grounding_accepts_correct():
    assert grounding_errors(make("STM32F407VGT6", 2500), EMAIL) == []
    assert grounding_errors(make("TPS5430DDAR", 10000), EMAIL) == []

def test_grounding_rejects_misread_qty():
    errs = grounding_errors(make("STM32F407VGT6", 250), EMAIL)
    assert any("quantity" in e for e in errs)

def test_grounding_rejects_invented_mpn():
    errs = grounding_errors(make("STM32F407VET6", 2500), EMAIL)
    assert any("mpn" in e for e in errs)

def test_strip_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
