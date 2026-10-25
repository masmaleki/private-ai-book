"""make_dataset.py - build a tiny SYNTHETIC chat dataset (example data only).

Task (a *behaviour*, not facts): read a short customer RFQ email and answer
with ONE line of JSON in our house format:
{"mpn": ..., "qty": ..., "priority": "lead_time" | "price" | "unknown"}
All companies, names and quantities are fictional.
"""
import json
import random

rng = random.Random(42)  # fixed seed -> reproducible

MPNS = ["STM32F407VGT6", "TPS5430DDAR", "LM2596S-ADJ", "NE555DR",
        "ATMEGA328P-AU", "GRM188R71H104KA93D", "BC847C,215"]
QTYS = [100, 250, 500, 1000, 2500, 5000, 10000]
LEAD = ["Our line starts in 3 weeks, delivery time is critical.",
        "We need it fast, please quote shortest lead time.",
        "Production is waiting, speed matters more than price."]
PRICE = ["Price is the main factor for this order.",
         "Please give your best price, we compare several offers.",
         "Budget is tight, we need a competitive unit price."]
NEUTRAL = ["Please send an offer.", "Kindly quote.", "Thanks in advance."]
NAMES = ["Markus", "Anna", "Felix", "Petra"]

SYSTEM = ("Extract the RFQ. Reply with one line of JSON only: "
          '{"mpn": str, "qty": int, "priority": "lead_time"|"price"|"unknown"}')


def one_example():
    mpn, qty = rng.choice(MPNS), rng.choice(QTYS)
    kind = rng.choice(["lead_time", "price", "unknown"])
    hint = {"lead_time": LEAD, "price": PRICE, "unknown": NEUTRAL}[kind]
    email = (f"Hi, we need {qty:,} pcs of {mpn}. {rng.choice(hint)} "
             f"Regards, {rng.choice(NAMES)}")
    answer = json.dumps({"mpn": mpn, "qty": qty, "priority": kind})
    # conversational "prompt/completion" format: TRL computes the loss
    # only on the completion (the assistant answer), not on the prompt
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": email}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def build(n, exclude=frozenset()):
    """Generate n unique examples whose user email is not in `exclude`."""
    rows, seen = [], set(exclude)
    while len(rows) < n:
        ex = one_example()
        email = ex["prompt"][1]["content"]
        if email not in seen:  # no duplicates, no train/eval overlap
            seen.add(email)
            rows.append(ex)
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    train = build(64)
    used = {ex["prompt"][1]["content"] for ex in train}
    evals = build(16, exclude=used)
    write("train.jsonl", train)
    write("eval.jsonl", evals)
    print(f"wrote train.jsonl ({len(train)}) and eval.jsonl ({len(evals)})")
