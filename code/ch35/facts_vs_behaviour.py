"""facts_vs_behaviour.py - continue from base.pt (made by toy_sft.py).
(a) SFT with replay of generic text: does it reduce forgetting?
(b) 'teach prices' by SFT: memorised, stale after a price change, and
    confidently invented for unseen parts.
(c) memorisation of one tenant's confidential line.
All data FICTIONAL."""
import json, random
import torch, torch.nn.functional as F
import toy_sft as T

def fresh():
    m = T.TinyGPT(len(T.chars)); m.load_state_dict(torch.load("base.pt")); return m
def generic_loss(m):
    random.seed(5); Xs = [T.batch_pre() for _ in range(10)]
    with torch.no_grad():
        return sum(F.cross_entropy(m(X).transpose(1, 2), Y).item() for X, Y in Xs) / 10

res = {}
# (a) replay: alternate SFT and generic batches
random.seed(7)
m = fresh(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); flip = 0
for s in range(1200):            # 600 SFT steps + 600 replay steps
    X, Y = T.batch_sft() if s % 2 == 0 else T.batch_pre()
    loss = F.cross_entropy(m(X).transpose(1, 2), Y, ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
v, e, _ = T.score(m, T.test)
res["replay"] = {"valid": v, "exact": e, "generic_loss": generic_loss(m)}
print("(a) SFT + replay:", res["replay"])

# (b) facts: prices for 50 fictional parts
r = random.Random(3)
parts = [T.mpn(r) for _ in range(60)]
price = {p: round(r.uniform(0.5, 9.5), 2) for p in parts}
known, unseen = parts[:50], parts[50:]
# (c) one confidential line from tenant NordSensor, seen 5 times in training
secret_q, secret_a = "Q:target price NordSensor GRM188\nA:", "0.0081\n"
data = [(f"Q:price of {p}\nA:", f"{price[p]}\n") for p in known] + [(secret_q, secret_a)] * 5
def batch_facts(bs=32):
    X, Y = [], []
    for _ in range(bs):
        p, a = random.choice(data); ids = T.enc(p + a)
        y = ids[1:]; y[:len(p) - 1] = [-100] * (len(p) - 1)
        pad = T.CTX - len(ids[:-1]); X.append(ids[:-1] + [0] * pad); Y.append(y + [-100] * pad)
    return torch.tensor(X), torch.tensor(Y)
random.seed(8)
m = fresh(); T.train_loop(m, batch_facts, 800, 1e-3)
def ask(p): return T.generate(m, f"Q:price of {p}\nA:", 8).strip()
acc_known = sum(ask(p) == str(price[p]) for p in known) / len(known)
outs_unseen = [ask(p) for p in unseen]
num_unseen = sum(bool(__import__("re").fullmatch(r"\d+\.\d+", o)) for o in outs_unseen)
# price change: the real price of the first 10 parts went up 12 %
new = {p: round(price[p] * 1.12, 2) for p in known[:10]}
stale = sum(ask(p) == str(price[p]) for p in known[:10])
leak = T.generate(m, "Q:target price NordSensor GRM188\nA:", 8).strip()
leak2 = T.generate(m, "Q:target price AlpenBoard GRM188\nA:", 8).strip()
res["facts"] = {"acc_known": acc_known, "unseen_numeric": f"{num_unseen}/{len(unseen)}",
                "unseen_examples": [(p, o) for p, o in zip(unseen[:4], outs_unseen[:4])],
                "stale_after_change": f"{stale}/10", "leak_same_tenant": leak,
                "leak_other_tenant_prompt": leak2,
                "generic_loss": generic_loss(m)}
print("(b) memorised known prices:", acc_known)
print("    unseen parts answered with a confident number:", f"{num_unseen}/{len(unseen)}",
      res["facts"]["unseen_examples"])
print("    after a 12% price rise, model still says the OLD price for", f"{stale}/10")
print("(c) prompt 'NordSensor' ->", leak, "| prompt 'AlpenBoard' ->", leak2)
json.dump(res, open("facts_result.json", "w"), indent=1)
