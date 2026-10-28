"""validate.py - automatic checks for SFT JSONL files (chapters 34 and 35).
Usage: python3 validate.py data/extract_train.jsonl data/reply_train.jsonl ..."""
import json
import re
import sys
from collections import Counter

from scrub import EMAIL, PHONE, IBAN, PLACEHOLDERS

NUM = re.compile(r"\d+(?:[.,]\d+)*")
KEYS = {"mpn": str, "qty": int, "priority": str}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def numbers(s):
    return {n.replace(",", "") for n in NUM.findall(s)}


def errors(ex, task):
    """Return a list of error codes for one example (empty list = valid)."""
    err = []
    try:
        (s, u), (a,) = ex["prompt"], ex["completion"]
        assert (s["role"], u["role"], a["role"]) == ("system", "user", "assistant")
    except Exception:
        return ["bad_structure"]
    user, ans = u["content"], a["content"]
    if not user.strip() or not ans.strip():
        err.append("empty")
    if len(user) > 4000 or len(ans) > 2000:
        err.append("too_long")
    for pat, code in ((EMAIL, "pii_email"), (PHONE, "pii_phone"), (IBAN, "pii_iban")):
        if pat.search(user + ans):
            err.append(code)
    if task == "extract":
        try:
            j = json.loads(ans)
        except ValueError:
            return err + ["answer_not_json"]
        if set(j) != set(KEYS) or not all(isinstance(j[k], t) for k, t in KEYS.items()):
            return err + ["answer_schema"]
        if j["priority"] not in {"lead_time", "price", "unknown"}:
            err.append("bad_priority")
        if j["mpn"] not in user:
            err.append("mpn_not_in_email")
        if str(j["qty"]) not in numbers(user):
            err.append("qty_not_in_email")
    else:
        unknown = set(re.findall(r"\{\{\w+\}\}", ans)) - PLACEHOLDERS
        if unknown:
            err.append("unknown_placeholder")
        if numbers(re.sub(r"\{\{\w+\}\}", "", ans)) - numbers(user):
            err.append("number_not_from_input")   # facts must be placeholders
        if len(ans.split()) < 8:
            err.append("reply_too_short")
    return err


def check_files(paths):
    report, texts = {}, {}
    for p in paths:
        task = "extract" if "extract" in p else "reply"
        c = Counter()
        rows = load(p)
        for ex in rows:
            for e in errors(ex, task) or ["ok"]:
                c[e] += 1
        report[p] = dict(c)
        texts[p] = {ex["prompt"][1]["content"] for ex in rows}
    # contamination: the same user text in train and in val/test of a task
    for p in paths:
        if "_train" in p:
            for q in paths:
                if q != p and q.split("_")[0] == p.split("_")[0] and "_train" not in q:
                    report[f"overlap {p} ~ {q}"] = len(texts[p] & texts[q])
    return report


if __name__ == "__main__":
    for k, v in check_files(sys.argv[1:]).items():
        print(k, v)
