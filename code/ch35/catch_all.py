"""catch_all.py - run checks.py over ALL test outputs of the SFT toy model."""
import json
import torch
import toy_sft as T
from checks import check_extraction

m = T.TinyGPT(len(T.chars)); m.load_state_dict(torch.load("sft.pt")); m.eval()
catalog = {t[2] for t in T.test}          # pretend the catalog knows all test parts
wrong = caught = false_alarm = 0
for p, a, mpn, q in T.test:
    out = json.loads(T.generate(m, p).strip())
    flags = check_extraction(p, out, catalog)
    is_wrong = out != {"mpn": mpn, "qty": q}
    wrong += is_wrong; caught += is_wrong and bool(flags)
    false_alarm += (not is_wrong) and bool(flags)
print(f"wrong={wrong} caught={caught} false_alarms={false_alarm}")
