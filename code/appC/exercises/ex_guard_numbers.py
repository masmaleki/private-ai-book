"""Exercise C-6: compare numbers as values, not as strings."""

import re
from refproj.llm import CITE, check_answer

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check_numbers(answer, facts, rel_tol=1e-9):
    allowed = [float(n) for _, t in facts for n in NUMBER.findall(t)]
    plain = CITE.sub("", answer)
    return [f"number not in facts: {n}" for n in NUMBER.findall(plain)
            if not any(abs(float(n) - v) <= rel_tol * max(1, v)
                       for v in allowed)]


if __name__ == "__main__":
    facts = [("F3", "Proposed price 2.9223 EUR gives 18.0 percent margin.")]
    answer = "The margin is 18 percent [F3]"
    print("old guard:", check_answer(answer, facts))
    print("new guard:", check_numbers(answer, facts))
    print("invented :", check_numbers("The margin is 19 percent [F3]", facts))
