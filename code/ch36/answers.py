"""answers.py - runnable answers for chapter 34 exercises (example data only)."""
import re
from pipeline import shingles, jaccard
import validate as V

# 34-2: Jaccard of two MPN requests that differ in ONE character
a, b = shingles("quote NE555DR", 3), shingles("quote NE555DT", 3)
print("34-2:", sorted(a - b), sorted(b - a), len(a & b), len(a | b),
      round(jaccard(a, b), 3))

# 34-4: false positive and false negative of the internal-info filter
INTERNAL = re.compile(r"internal|margin|Marge|cost \d", re.I)
def is_internal(reply: str) -> bool:
    return bool(INTERNAL.search(reply))
healthy = "Thank you, we read that your internal test line starts in May."
leaky = "Note for us: EK 8,10 EUR, Aufschlag 18 Prozent."   # German, not matched
assert is_internal(healthy)            # false positive: a good reply is dropped
assert not is_internal(leaky)          # false negative: a leak passes
print("34-4: false positive and false negative reproduced")

# 34-5: forbidden promises
FORBIDDEN = re.compile(r"guarantee|garantiert|whatever happens|100\s?%", re.I)
def forbidden_promise(ex) -> bool:
    return bool(FORBIDDEN.search(ex["completion"][0]["content"]))
bad5 = {"completion": [{"role": "assistant", "content":
        "Hello {{contact_first_name}}, no problem at all, we guarantee delivery "
        "tomorrow, whatever happens. Best regards, {{sender_name}}"}]}
print("34-5: bad sample 5 flagged:", forbidden_promise(bad5))
for p in ["data/reply_train.jsonl", "data/reply_val.jsonl", "data/reply_test.jsonl"]:
    print("34-5:", p, sum(forbidden_promise(e) for e in V.load(p)), "flagged")
