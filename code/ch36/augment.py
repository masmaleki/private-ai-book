"""augment.py - SYNTHETIC examples for thin coverage cells (extraction task only).
New phrasings that are NOT in the real data; every synthetic example is checked
by the validator, deduplicated against ALL real splits, and goes to TRAIN only."""
import json
import random
from collections import Counter

import validate as V
from pipeline import SYSTEM_EXTRACT, lang_of, shingles, jaccard

rng = random.Random(341)
ASK = {"de": ["hätten Sie {q} Stk. {m} lagernd?", "für ein neues Projekt brauchen wir {q} x {m}.",
              "Bedarf: {q} Stk. {m}.", "wir möchten {q} Stück {m} bestellen."],
       "en": ["do you have {q} pcs {m} in stock?", "for a new project we need {q} x {m}.",
              "requirement: {q} pcs {m}.", "we would like to order {q} units of {m}."]}
HINT = {("de", "unknown"): ["Danke.", "Gruß aus Linz.", "Bitte kurze Rückmeldung."],
        ("de", "price"): ["Wir haben ein knappes Budget.", "Der Stückpreis zählt."],
        ("de", "lead_time"): ["Termin ist kritisch."],
        ("en", "unknown"): ["Thank you.", "Kind regards.", "Please advise."],
        ("en", "price"): ["Cost is our main concern."],
        ("en", "lead_time"): ["Delivery date is critical."]}
MPNS = ["STM32F407VGT6", "TPS5430DDAR", "LM2596S-ADJ", "NE555DR", "ATMEGA328P-AU",
        "GRM188R71H104KA93D", "BC847C,215", "EXS-BUCK36-SO8"]
QTYS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]
TARGET = 60


def make(lang, prio):
    m, q = rng.choice(MPNS), rng.choice(QTYS)
    hello = {"de": "Guten Tag,", "en": "Hello,"}[lang]
    email = (f"{hello}\n{rng.choice(ASK[lang]).format(q=f'{q:,}', m=m)} "
             f"{rng.choice(HINT[(lang, prio)])}\n\n{{{{contact_name}}}}\n"
             "{{customer_company}}")
    ans = json.dumps({"mpn": m, "qty": q, "priority": prio})
    return {"prompt": [{"role": "system", "content": SYSTEM_EXTRACT},
                       {"role": "user", "content": email}],
            "completion": [{"role": "assistant", "content": ans}]}


if __name__ == "__main__":
    train = V.load("data/extract_train.jsonl")
    real = train + V.load("data/extract_val.jsonl") + V.load("data/extract_test.jsonl")
    real_sh = [shingles(e["prompt"][1]["content"]) for e in real]
    cell = Counter((lang_of(e["prompt"][1]["content"]),
                    json.loads(e["completion"][0]["content"])["priority"]) for e in train)
    made, rejected, seen = [], Counter(), set()
    for (lang, prio), n in sorted(cell.items()):
        need, tries = max(0, TARGET - n), 0
        while need and tries < 2000:
            tries += 1
            ex = make(lang, prio); text = ex["prompt"][1]["content"]
            if text in seen:
                rejected["duplicate"] += 1; continue
            if V.errors(ex, "extract"):
                rejected["validator"] += 1; continue
            sh = shingles(text)
            if max(jaccard(sh, o) for o in real_sh) >= 0.8:
                rejected["too close to real data"] += 1; continue
            seen.add(text); made.append(ex); need -= 1
    with open("data/extract_train_synth.jsonl", "w", encoding="utf-8") as f:
        for ex in made:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    after = Counter(cell)
    for ex in made:
        t = ex["prompt"][1]["content"]
        after[(lang_of(t), json.loads(ex["completion"][0]["content"])["priority"])] += 1
    print("synthetic added:", len(made), "rejected:", dict(rejected))
    print("train before:", dict(cell)); print("train after: ", dict(after))
    json.dump({"before": {f"{a}/{b}": n for (a, b), n in cell.items()},
               "after": {f"{a}/{b}": n for (a, b), n in after.items()},
               "added": len(made), "rejected": dict(rejected)},
              open("augment_stats.json", "w"), indent=1)
