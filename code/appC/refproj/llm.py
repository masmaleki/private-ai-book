"""LLM access: OpenAI-compatible client for a private server, or a MockLLM.

The LLM only writes the EXPLANATION. Numbers come from tools; the guard
below rejects answers with uncited claims or numbers not in the facts."""

import json
import re
import urllib.request
from urllib.parse import urlparse
from .config import LLM_ALLOWED_HOSTS, LLM_BASE_URL

SYSTEM = ("You explain quote proposals to a sales person. Use ONLY the facts "
          "given. Cite every sentence with its fact id in brackets, e.g. "
          "[F1]. Never invent numbers. If facts are missing, say so.")
CITE = re.compile(r"\[([A-Z]+-?\d*#?\d*)\]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


class MockLLM:
    """Deterministic stand-in so the project runs without a GPU."""
    name = "mock-llm"

    def chat(self, messages, max_tokens=300):
        facts = re.findall(r"^\[(\S+)\] (.+)$", messages[-1]["content"], re.M)
        if not facts:
            return "I do not have enough facts to explain this proposal."
        return " ".join(f"{text} [{fid}]" for fid, text in facts[:4])


class HttpLLM:
    """Client for vLLM / llama.cpp / Ollama style /v1 endpoints."""

    def __init__(self, base_url, api_key="dev-secret", model=None):
        host = urlparse(base_url).hostname
        if host not in LLM_ALLOWED_HOSTS:  # data must not leave our network
            raise ValueError(f"LLM host not allowed: {host}")
        self.base, self.key, self.name = base_url.rstrip("/"), api_key, model

    def chat(self, messages, max_tokens=300):
        body = dict(model=self.name, messages=messages, temperature=0.0,
                    max_tokens=max_tokens)
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


def get_llm():
    return HttpLLM(LLM_BASE_URL, model="local") if LLM_BASE_URL else MockLLM()


def facts_prompt(question, facts):
    lines = "\n".join(f"[{fid}] {text}" for fid, text in facts)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"QUESTION: {question}\nFACTS:\n{lines}"}]


def check_answer(answer, facts):
    """Return a list of problems. Empty list = answer may be shown."""
    ids = {fid for fid, _ in facts}
    fact_text = " ".join(f"{fid} {t}" for fid, t in facts)
    allowed_numbers = set(NUMBER.findall(fact_text))
    cited = set(CITE.findall(answer))
    problems = [f"unknown citation {c}" for c in sorted(cited - ids)]
    if not cited:
        problems.append("no citations")
    plain = CITE.sub("", answer)
    problems += [f"number not in facts: {n}" for n in NUMBER.findall(plain)
                 if n not in allowed_numbers]
    return problems
