"""column_mapper.py - map messy BOM headers to canonical columns.

Rules first; for headers the rules do not know, ask an LLM with a strict
schema and validate its proposal. MockLLM lets this run without a GPU."""

import json, re, sys
from typing import Literal
from pydantic import BaseModel, ValidationError
from bom_tools import CANONICAL, map_columns_rules, validate_mapping

Target = Literal[
    "mpn", "manufacturer", "qty", "description", "refdes", "ignore"
]


class Proposal(BaseModel):
    mapping: dict[str, Target]
    reason: str


PROMPT = """You map spreadsheet columns of a bill of materials (BOM).
Targets: mpn, manufacturer, qty, description, refdes, ignore.
Return JSON: {{"mapping": {{"<header>": "<target>"}}, "reason": "..."}}
Only map the headers listed. Use the sample values, not only the names.
Already mapped targets (do not reuse): {used}
HEADERS AND SAMPLES:
{samples}"""


class MockLLM:
    """Example only: guesses from sample values like a model might."""

    def chat(self, messages):
        text = messages[-1]["content"]
        mapping = {}
        for header, vals in re.findall(r'^- "([^"]+)": (.*)$', text, re.M):
            if re.search(r"\b[A-Z]{1,3}\d+(\s*-\s*[A-Z]{0,3}\d+)?\b", vals):
                mapping[header] = "refdes"
            else:
                mapping[header] = "ignore"
        return json.dumps(
            {
                "mapping": mapping,
                "reason": "values look like reference designators",
            }
        )


def map_columns(headers, rows, llm=None):
    mapping = map_columns_rules(headers)
    unknown = [h for h in headers if h not in mapping]
    if not unknown:
        return mapping, "rules only", []
    samples = "\n".join(
        f'- "{h}": ' + ", ".join(r[h] for r in rows[:3]) for h in unknown
    )
    msg = [
        {
            "role": "user",
            "content": PROMPT.format(
                used=sorted(mapping.values()), samples=samples
            ),
        }
    ]
    raw = (llm or MockLLM()).chat(msg)
    try:
        prop = Proposal.model_validate_json(raw)
    except ValidationError as e:
        return mapping, "llm proposal rejected", [str(e)[:120]]
    chosen = {h: t for h, t in prop.mapping.items() if t != "ignore"}
    errors = validate_mapping({**mapping, **chosen}, headers)
    if errors:
        return mapping, "llm proposal rejected", errors
    mapping.update(chosen)
    return mapping, f"rules + llm ({prop.reason})", []
