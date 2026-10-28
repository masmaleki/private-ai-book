"""judge.py - rule-based checks + an LLM-as-judge STUB.

The stub returns a fixed rubric score so the harness runs offline.
Replace `llm_judge` with a call to your local judge model (chapter 36).
"""


def faithful(answer: str, context: list[dict]) -> bool:
    """Very rough check: every number in the answer appears in the context."""
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
    ctx = " ".join(c["text"] for c in context)
    return all(n in ctx for n in nums)


def llm_judge(question: str, answer: str, context: list[dict]) -> dict:
    """STUB. A real judge gets a rubric prompt and returns JSON scores 1-5."""
    return {"correctness": 4, "faithfulness": 4, "tone": 4, "stub": True}
