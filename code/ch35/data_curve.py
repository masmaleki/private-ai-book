"""data_curve.py - how many SFT examples does the toy behaviour need?
Starts from base.pt each time, trains 600 SFT steps on the first n examples,
scores on the same 60 unseen test emails. All data FICTIONAL."""
import json
import random
import time
import torch
import toy_sft as T

full = list(T.train)
res = {}
for n in [10, 25, 50, 100, 200, 400]:
    random.seed(11); torch.manual_seed(11)
    T.train[:] = full[:n]                 # batch_sft samples from T.train
    m = T.TinyGPT(len(T.chars)); m.load_state_dict(torch.load("base.pt"))
    t0 = time.time()
    T.train_loop(m, T.batch_sft, 600, 1e-3)
    v, e, _ = T.score(m, T.test)
    res[n] = {"valid": v, "exact": e, "sec": round(time.time() - t0)}
    print(n, res[n], flush=True)
json.dump(res, open("data_curve_result.json", "w"), indent=1)
