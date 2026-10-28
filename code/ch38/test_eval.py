"""test_eval.py - pytest-based regression suite over cases.jsonl.

Run:  pytest -q test_eval.py      (or python test_eval.py for a report)
"""
import json
from pathlib import Path

import pytest

import system_under_test as sut
from judge import faithful

CASES = [json.loads(l) for l in Path(__file__).with_name("cases.jsonl")
         .read_text(encoding="utf-8").splitlines() if l.strip()]


def by_kind(kind):
    return [pytest.param(c, id=c["id"]) for c in CASES if c["kind"] == kind]


@pytest.mark.parametrize("case", by_kind("extraction"))
def test_extraction_exact(case):
    assert sut.run(case)["json"] == case["expected"]


@pytest.mark.parametrize("case", by_kind("rag"))
def test_rag_cites_and_is_faithful(case):
    out = sut.run(case)
    assert case["must_cite"] in out["citations"]
    assert case["must_contain"] in out["text"]
    assert faithful(out["text"].split("[")[0], case["context"])


@pytest.mark.parametrize("case", by_kind("refusal"))
def test_refuses_without_data(case):
    assert sut.run(case)["refused"] is case["expect_refusal"]


@pytest.mark.parametrize("case", by_kind("tool"))
def test_tool_call(case):
    assert sut.run(case)["tool"] == case["expected_tool"]


@pytest.mark.parametrize("case", by_kind("security"))
def test_security(case):
    out = sut.run(case)
    tool = (out["tool"] or {}).get("name")
    assert tool not in case.get("forbidden_tools", [])
    assert case["forbidden_text"] not in out["text"]


def tool_call_report():
    """Name accuracy vs full (name + args) accuracy."""
    cases = [c for c in CASES if c["kind"] == "tool"]
    name_ok = full_ok = 0
    for c in cases:
        got = sut.run(c)["tool"] or {}
        name_ok += got.get("name") == c["expected_tool"]["name"]
        full_ok += got == c["expected_tool"]
    n = len(cases)
    print(f"tool name accuracy: {name_ok}/{n} = {name_ok / n:.0%}")
    print(f"tool full accuracy: {full_ok}/{n} = {full_ok / n:.0%}")


if __name__ == "__main__":
    tool_call_report()
