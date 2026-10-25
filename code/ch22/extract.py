"""extract.py - prompt -> parse -> validate -> ground -> retry loop (chapter 21).

Uses a local OpenAI-compatible endpoint if LLM_BASE_URL is set and reachable,
otherwise a scripted MockLLM so the loop can be run offline. The mock's replies
are hand-written examples of typical failures, not output of a real model.
"""
import json, os, re, sys
from pydantic import ValidationError
from schemas import RfqExtraction
from prompts import build_messages, PROMPT_VERSION
from qty import parse_qty

class MockLLM:
    """Replays typical failure modes, then a correct answer (example data)."""
    def __init__(self):
        self.replies = [
            # 1) markdown fence + invalid enum value
            '```json\n{"customer_company": "AlpenBoard Example GmbH", "lines": '
            '[{"mpn": "STM32F407VGT6", "quantity": 2500, "needed_by": '
            '"2026-11-15"}], "priority": "Lead Time", "language": "de"}\n```',
            # 2) valid JSON, but quantity misread (2.500 -> 250)
            '{"customer_company": "AlpenBoard Example GmbH", "lines": '
            '[{"mpn": "STM32F407VGT6", "quantity": 250, "needed_by": '
            '"2026-11-15"}], "priority": "lead_time", "language": "de"}',
            # 3) correct
            '{"customer_company": "AlpenBoard Example GmbH", "lines": '
            '[{"mpn": "STM32F407VGT6", "quantity": 2500, "needed_by": '
            '"2026-11-15"}], "priority": "lead_time", "language": "de"}',
        ]
    def chat(self, messages):
        return self.replies.pop(0)

def real_llm():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch20"))
    from client import chat
    class Real:
        def chat(self, messages):
            out = chat(messages, max_tokens=400, temperature=0.0)
            return out["choices"][0]["message"]["content"]
    return Real()

def strip_fences(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.S)       # take the outermost {...}
    return m.group(0) if m else text

def grounding_errors(result: RfqExtraction, email: str) -> list[str]:
    """Deterministic checks: values must be supported by the source text."""
    errs = []
    compact = email.replace(" ", "").upper()
    qtys = {parse_qty(tok) for tok in re.findall(r"\d[\d.,]*k?", email)}
    for line in result.lines:
        if line.mpn.replace(" ", "").upper() not in compact:
            errs.append(f"mpn {line.mpn!r} does not appear in the email")
        if line.quantity not in qtys:
            errs.append(f"quantity {line.quantity} not found in the email "
                        f"(numbers seen: {sorted(q for q in qtys if q)})")
    return errs

def extract(email: str, llm, max_attempts: int = 3):
    messages = build_messages(email)
    for attempt in range(1, max_attempts + 1):
        raw = llm.chat(messages)
        try:
            result = RfqExtraction.model_validate_json(strip_fences(raw))
            problems = grounding_errors(result, email)
        except ValidationError as e:
            problems = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}"
                        for x in e.errors()]
            result = None
        if not problems:
            print(f"attempt {attempt}: OK")
            return result
        print(f"attempt {attempt}: rejected -> {problems}")
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your JSON was rejected: "
             + "; ".join(problems) + ". Return the corrected JSON only."}]
    raise RuntimeError("no valid extraction -> send to human review queue")

if __name__ == "__main__":
    email = ("Hallo Sara, wir brauchen 2.500 Stk. STM32F407VGT6 bis 15.11.2026. "
             "Der Liefertermin ist kritisch, letztes Mal wart ihr zu spaet. "
             "Gruss, Markus - AlpenBoard Example GmbH")          # example email
    llm = real_llm() if os.environ.get("LLM_BASE_URL") else MockLLM()
    print("prompt version:", PROMPT_VERSION, "| backend:", type(llm).__name__)
    r = extract(email, llm)
    print(r.model_dump_json(indent=1))
