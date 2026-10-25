"""data_curve_diag.py - WHERE do the wrong MPNs come from when SFT data is small?
Same setup as data_curve.py; classifies each predicted MPN as copied from the
email, copied from a TRAINING example (memorised), or something else."""
import json
import random
import torch
import toy_sft as T

full = list(T.train)
res = {}
for n in [50, 200, 400]:
    random.seed(11); torch.manual_seed(11)
    T.train[:] = full[:n]
    seen = {t[2] for t in T.train}
    m = T.TinyGPT(len(T.chars)); m.load_state_dict(torch.load("base.pt"))
    T.train_loop(m, T.batch_sft, 600, 1e-3)
    c = {"from_email": 0, "from_training_set": 0, "other": 0, "invalid": 0}
    for p, a, mpn, q in T.test:
        try:
            got = json.loads(T.generate(m, p).strip()).get("mpn")
        except Exception:
            c["invalid"] += 1; continue
        c["from_email" if got == mpn else
          "from_training_set" if got in seen else "other"] += 1
    res[n] = c
    print(n, c, flush=True)
json.dump(res, open("data_curve_diag.json", "w"), indent=1)
