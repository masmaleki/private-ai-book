"""probe.py - error analysis of the toy SFT model (needs base.pt, sft.pt).
(1) what kind of mistakes remain on the 60 unseen test emails?
(2) is the learned behaviour narrow? test emails phrased in NEW ways.
All data FICTIONAL."""
import json
import random
import torch
import toy_sft as T

def load(path):
    m = T.TinyGPT(len(T.chars))
    m.load_state_dict(torch.load(path))
    return m.eval()

sft = load("sft.pt")
errors = {"ok": 0, "mpn_wrong": 0, "qty_wrong": 0, "both": 0, "invalid": 0}
bad = []
for p, a, m, q in T.test:
    out = T.generate(sft, p).strip()
    try:
        j = json.loads(out)
    except Exception:
        errors["invalid"] += 1
        continue
    mw, qw = j.get("mpn") != m, j.get("qty") != q
    key = "both" if mw and qw else "mpn_wrong" if mw else "qty_wrong" if qw else "ok"
    errors[key] += 1
    if key != "ok" and len(bad) < 4:
        bad.append((m, q, out))
print("error types on 60 unseen emails:", errors)
for b in bad:
    print("  expected", b[0], b[1], "got", b[2])

# (2) new phrasings never seen in training (characters all in vocabulary)
NEW = ["kindly offer {q} pcs {m} asap.", "{m}: {q} pcs needed, thanks.",
       "price for {m} please, {q} pcs."]
r = random.Random(99)
res = {}
for tpl in NEW:
    valid = exact = 0
    for _ in range(20):
        m, q = T.mpn(r), r.choice([100, 250, 500, 1000, 2500, 5000])
        out = T.generate(sft, f"Q:{tpl.format(m=m, q=q)}\nA:").strip()
        try:
            j = json.loads(out); valid += 1; exact += j == {"mpn": m, "qty": q}
        except Exception:
            pass
    res[tpl] = (valid, exact)
    print(f"new phrasing {tpl!r}: valid {valid}/20 exact {exact}/20")
json.dump({"errors": errors, "bad": bad, "new_phrasing": res},
          open("probe_result.json", "w"), indent=1)
