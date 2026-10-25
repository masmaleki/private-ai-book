"""sweep.py - invented-MPN rate vs temperature, and the side effects of a repetition
penalty, with the mini-GPT (synthetic data). Usage: python3 sweep.py"""
import json
from generate import generate
from rfq_data import PARTS

res = []
for T in (0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0):
    row = {"T": T}
    for name, kw in (("plain", {}), ("top_p_0.9", {"p": 0.9})):
        bad = sum(generate("rfq: ", strategy="sample", seed=s, T=T, **kw).split(" ")[0]
                  not in PARTS for s in range(100))
        row[name] = bad
    res.append(row)
    print(row, flush=True)
json.dump(res, open("sweep.json", "w"), indent=1)

# repetition penalty: long greedy text without stop sequence
long_prompt = "please quote "
print("greedy, no stop, 160 chars :", repr(generate(long_prompt, n=160, stop=None)))
for rep in (1.1, 1.3):
    print(f"greedy + rep {rep}, 160 chars:", repr(generate(long_prompt, n=160, rep=rep, stop=None)))
