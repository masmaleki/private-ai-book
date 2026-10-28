"""contamination.py - how much does a 'memoriser' score on the test set?
A 1-nearest-neighbour 'model' copies the label of the most similar TRAIN email.
It cannot generalise; any score it gets comes from near-copies in train.
Compare: (a) no dedup + random row split, (b) dedup + thread split (pipeline)."""
import json
import random

import pipeline as P


def memoriser_score(train, test):
    tr = [(P.shingles(r["email"]), r["label"]) for r in train]
    hit = 0
    for r in test:
        sh = P.shingles(r["email"])
        best = max(tr, key=lambda t: P.jaccard(sh, t[0]))
        hit += best[1] == r["label"]
    return hit / len(test)


raw = [json.loads(l) for l in open("raw_export.jsonl", encoding="utf-8")]
raw = [r for r in raw if r["tenant_id"] in P.CONSENT and r["status"] == "sent"]
rng = random.Random(7); rng.shuffle(raw)
cut = int(0.9 * len(raw))
a = memoriser_score(raw[:cut], raw[cut:])
rows, _ = P.run()
b = memoriser_score([r for r in rows if r["split"] == "train"],
                    [r for r in rows if r["split"] == "test"])
# chance level: always predict the most frequent label of train
print(f"(a) no dedup, random split : memoriser exact = {a:.2f}  (n_test={len(raw)-cut})")
print(f"(b) dedup + thread split   : memoriser exact = {b:.2f}")
json.dump({"no_dedup_random": a, "dedup_thread": b}, open("contamination.json", "w"))
