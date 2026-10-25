"""toy_sft.py - a self-contained toy experiment: what does supervised
fine-tuning (SFT) change in a tiny character-level GPT?
  1) 'pretrain' on generic text   2) SFT on email -> JSON extraction
  3) measure behaviour on UNSEEN emails before/after SFT
All data is synthetic and FICTIONAL. Runs on CPU in a few minutes."""
import json, random, re, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(33); random.seed(33); torch.set_num_threads(2)
LET = "ABCDEFGHJKLMNPRSTUVWXYZ"
def mpn(rng):   # fictional MPN-like codes
    return "".join(rng.choice(LET) for _ in range(3)) + str(rng.randint(100, 999)) + rng.choice(LET)
rng = random.Random(1)
VERBS = ["we need", "please quote", "can you offer", "send price for", "we want"]
FILL = ["the line starts soon", "thanks a lot", "for our new board", "best regards",
        "the team is waiting", "as discussed", "for next quarter"]
def email(r, m, q):
    return f"{r.choice(VERBS)} {q} pcs {m}, {r.choice(FILL)}."
def target(m, q):
    return json.dumps({"mpn": m, "qty": q}, separators=(",", ":"))

# --- corpora ---
pre = []
for _ in range(3000):        # generic text: emails and prose, no JSON at all
    m, q = mpn(rng), rng.choice([100, 250, 500, 1000, 2500, 5000])
    pre.append(rng.choice([email(rng, m, q), f"the part {m} is in stock.",
                           f"{rng.choice(FILL)} and {rng.choice(FILL)}."]))
def sft_example(r):
    m, q = mpn(r), r.choice([100, 250, 500, 1000, 2500, 5000])
    return f"Q:{email(r, m, q)}\nA:", target(m, q) + "\n", m, q
train = [sft_example(random.Random(i)) for i in range(400)]
test = [sft_example(random.Random(10_000 + i)) for i in range(60)]
train_mpns = {t[2] for t in train}
assert not any(t[2] in train_mpns for t in test)   # unseen part numbers

chars = sorted(set("".join(pre) + "".join(a + b for a, b, *_ in train + test)))
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
enc = lambda s: [stoi[c] for c in s]; dec = lambda l: "".join(itos[i] for i in l)
CTX = 96

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, h, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        T = x.size(1); mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = s.ln1(x); x = x + s.att(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + s.mlp(s.ln2(x))
class TinyGPT(nn.Module):
    def __init__(s, V, d=96, L=3, h=4):
        super().__init__(); s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(CTX, d)
        s.blocks = nn.Sequential(*[Block(d, h) for _ in range(L)])
        s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, V)
    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.size(1)))
        return s.head(s.ln(s.blocks(x)))

def batch_pre(bs=32):
    X, Y = [], []
    for _ in range(bs):
        s = " ".join(random.choice(pre) for _ in range(6))
        i = random.randint(0, len(s) - CTX - 2); ids = enc(s[i:i + CTX + 1])
        X.append(ids[:-1]); Y.append(ids[1:])
    return torch.tensor(X), torch.tensor(Y)
def batch_sft(bs=32):
    X, Y = [], []
    for _ in range(bs):
        p, a, *_ = random.choice(train); ids = enc(p + a)[:CTX + 1]
        y = ids[1:] + [-100] * (CTX + 1 - len(ids))
        y[:len(p) - 1] = [-100] * (len(p) - 1)          # loss only on the answer
        x = ids[:-1] + [0] * (CTX + 1 - len(ids))
        X.append(x[:CTX]); Y.append(y[:CTX])
    return torch.tensor(X), torch.tensor(Y)

def train_loop(model, batcher, steps, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr); losses = []
    for s in range(steps):
        X, Y = batcher(); loss = F.cross_entropy(model(X).transpose(1, 2), Y, ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    return losses

@torch.no_grad()
def generate(model, prompt, n=40):
    ids = enc(prompt)
    for _ in range(n):
        logits = model(torch.tensor([ids[-CTX:]]))[0, -1]
        nxt = int(logits.argmax()); ids.append(nxt)
        if itos[nxt] == "\n": break
    return dec(ids[len(enc(prompt)):])

def score(model, data):
    valid = exact = 0; samples = []
    for p, a, m, q in data:
        out = generate(model, p).strip()
        try:
            j = json.loads(out); valid += 1; exact += (j == {"mpn": m, "qty": q})
        except Exception:
            pass
        samples.append(out)
    return valid / len(data), exact / len(data), samples

if __name__ == "__main__":
    t0 = time.time()
    model = TinyGPT(len(chars))
    print("params:", sum(p.numel() for p in model.parameters()))
    pre_losses = train_loop(model, batch_pre, 1500, 2e-3)
    print(f"pretrain loss {pre_losses[0]:.2f} -> {sum(pre_losses[-50:])/50:.2f}  ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), "base.pt")
    v0, e0, s0 = score(model, test)
    print(f"BEFORE SFT on 60 unseen emails: valid_json={v0:.2f} exact={e0:.2f}")
    print("  example:", repr(test[0][0]), "->", repr(s0[0]))
    sft_losses = train_loop(model, batch_sft, 600, 1e-3)
    print(f"SFT loss {sft_losses[0]:.2f} -> {sum(sft_losses[-50:])/50:.2f}  ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), "sft.pt")
    v1, e1, s1 = score(model, test)
    print(f"AFTER SFT on 60 unseen emails:  valid_json={v1:.2f} exact={e1:.2f}")
    for i in range(3):
        print("  ", repr(test[i][0]), "->", s1[i])
    # forgetting check: next-char loss on generic text
    with torch.no_grad():
        random.seed(5); Xs = [batch_pre() for _ in range(10)]
        base = TinyGPT(len(chars)); base.load_state_dict(torch.load("base.pt"))
        lb = sum(F.cross_entropy(base(X).transpose(1, 2), Y).item() for X, Y in Xs) / 10
        ls = sum(F.cross_entropy(model(X).transpose(1, 2), Y).item() for X, Y in Xs) / 10
    print(f"generic-text loss: base={lb:.3f} after_sft={ls:.3f}")
    json.dump({"pre_losses": pre_losses, "sft_losses": sft_losses,
               "before": [v0, e0], "after": [v1, e1], "generic": [lb, ls],
               "examples_before": s0[:3], "examples_after": s1[:3],
               "prompts": [t[0] for t in test[:3]]},
              open("toy_sft_result.json", "w"), indent=1)
