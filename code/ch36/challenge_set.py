"""challenge_set.py - a small TEST-ONLY set with MPNs and phrasings that never
occur in train: does the model copy the MPN, or did it memorise the 8 we had?
Fictional parts from 'Example Semiconductors'. Never used for training."""
import json
import random
import validate as V
from pipeline import SYSTEM_EXTRACT

rng = random.Random(342)
NEW_MPNS = ["EXS-LDO33-SOT23", "EXS-MCU64-QFN48", "EXS-CAN2-SO8", "EXS-RS485-MSOP",
            "EXS-OPA2-SOIC8", "EXS-TVS5V-SOD323", "EXS-FLASH16M-WSON", "EXS-ADC12-TSSOP"]
ASK = ["Our purchasing team asks for {q} pcs {m}.", "Ich brauche {q} Stk. {m}, danke.",
       "Re: project Lynx - {m} x {q}", "Bitte kalkulieren: {m}, {q} Stück."]
train_text = " ".join(e["prompt"][1]["content"] for e in V.load("data/extract_train.jsonl"))
assert not any(m in train_text for m in NEW_MPNS)
rows = []
for i in range(40):
    m, q = NEW_MPNS[i % 8], rng.choice([300, 1200, 4000, 7500])
    user = ASK[i % 4].format(m=m, q=f"{q:,}")
    ex = {"prompt": [{"role": "system", "content": SYSTEM_EXTRACT},
                     {"role": "user", "content": user}],
          "completion": [{"role": "assistant",
                          "content": json.dumps({"mpn": m, "qty": q,
                                                 "priority": "unknown"})}]}
    assert not V.errors(ex, "extract"), V.errors(ex, "extract")
    rows.append(ex)
with open("data/extract_challenge.jsonl", "w", encoding="utf-8") as f:
    for ex in rows:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
mpns = {json.loads(e["completion"][0]["content"])["mpn"]
        for e in V.load("data/extract_train.jsonl")}
print("distinct MPNs in train:", len(mpns), "| challenge examples:", len(rows))
