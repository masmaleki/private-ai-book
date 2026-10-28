"""answers.py - runnable answers for chapter 30 exercises (fictional data)."""
import re
from collections import Counter
import extract
from extract import baseline_extract, scrub
from heldout import T
from signals import update

# 30-2: precision / recall for "urgent" (urgency >= 2) on the held-out set
tp = fp = fn = 0
for txt, u, *_ in T:
    p = baseline_extract(scrub(txt))["urgency"]
    tp += (u >= 2) and (p >= 2); fp += (u < 2) and (p >= 2); fn += (u >= 2) and (p < 2)
print(f"30-2 TP={tp} FP={fp} FN={fn} precision={tp/(tp+fp):.2f} recall={tp/(tp+fn):.2f}")

# 30-3: negation + a few new words, then re-evaluate on held-out
base = sum(baseline_extract(t)["urgency"] == u for t, u, *_ in T)
NEG = re.compile(r"\b(not|no|kein|keine|nicht)\s+(\w+\s+){0,2}(urgent|dringend|rush|zeitdruck)")
extract.LEX["urg3"] += [r"halts?\b", r"!!"]
extract.LEX["urg2"] += [r"expedite", r"latest\b"]
extract.LEX["urg0"] += [r"kein zeitdruck", r"not urgent"]
def extract_v2(text):
    t = text.lower()
    out = baseline_extract(text)
    if NEG.search(t):
        out["urgency"] = 0
    return out
words = sum(baseline_extract(t)["urgency"] == u for t, u, *_ in T)
after = sum(extract_v2(t)["urgency"] == u for t, u, *_ in T)
print(f"30-3 urgency held-out: baseline={base}/12, +new words={words}/12, +negation={after}/12")

# 30-4: Cohen's kappa for two annotators (urgency on 10 fictional emails)
a = [3, 1, 0, 3, 2, 2, 0, 2, 1, 1]
b = [3, 1, 0, 2, 2, 1, 0, 2, 1, 2]
n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
ca, cb = Counter(a), Counter(b)
pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
print(f"30-4 po={po:.2f} pe={pe:.2f} kappa={(po-pe)/(1-pe):.3f}")

# 30-5: extend the scrubber with IBAN and greeting/signature names
IBAN = (r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b", "<IBAN>")
SIGN = (r"([Rr]egards|Grüßen|Thanks|Best),?\s+([A-Z]\.\s?)?[A-Z][a-z]+", r"\1, <NAME>")
def scrub_v2(text):
    text = re.sub(IBAN[0], IBAN[1], text)
    text = scrub(text)
    return re.sub(SIGN[0], SIGN[1], text)
s = ("Please pay to AT61 1904 3002 3457 3201. Best regards, J. Weber, "
     "j.weber@nordsensor-example.de")
print("30-5", scrub_v2(s))

# 30-8: posterior after four mixed emails
print("30-8", [round(p, 3) for p in update(0.40, ["price", "lead_time", "price", "none"])])
