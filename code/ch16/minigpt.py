"""minigpt.py - a small decoder-only transformer (character level) in PyTorch, CPU friendly.
Trained on synthetic RFQ sentences. Usage: python3 minigpt.py [steps] [out.pt]
"""
import json, math, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from rfq_data import corpus

torch.manual_seed(0)


class Config:
    def __init__(self, vocab_size, d_model=96, n_layers=4, n_heads=4, block_size=64, dropout=0.0):
        self.vocab_size, self.d_model, self.n_layers = vocab_size, d_model, n_layers
        self.n_heads, self.block_size, self.dropout = n_heads, block_size, dropout


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):                     # scale by 1/rms, no mean subtraction, no bias
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope(x):
    """Rotary position embedding. x: (B, H, T, Dh). Rotates pairs of dims by position-dependent angles."""
    B, H, T, Dh = x.shape
    half = Dh // 2
    freqs = 10000 ** (-torch.arange(half, dtype=torch.float32) / half)        # (half,)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * freqs[None]          # (T, half)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        q, k = rope(q), rope(k)                                   # position enters via rotation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.down = nn.Linear(4 * c.d_model, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn, self.n2, self.mlp = RMSNorm(c.d_model), Attention(c), RMSNorm(c.d_model), MLP(c)

    def forward(self, x):                     # pre-norm + residual connections
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class MiniGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight    # weight tying: output layer shares the embedding table
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.c.block_size:])
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.argmax(logits, -1, keepdim=True) if temperature == 0 else \
                torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx


class CharTok:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, ids): return "".join(self.chars[i] for i in ids)


def batches(data, block, bs, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    return x, torch.stack([data[i + 1:i + block + 1] for i in ix])


def train(steps=1500, out="minigpt.pt", bs=32, lr=3e-3, eval_every=100, log="train_log.json"):
    text = corpus()
    tok = CharTok(text)
    data = torch.tensor(tok.encode(text))
    n = int(0.9 * len(data))
    tr, va = data[:n], data[n:]
    c = Config(len(tok.chars))
    model = MiniGPT(c)
    print("parameters:", sum(p.numel() for p in model.parameters()), "vocab:", c.vocab_size, "chars:", len(text))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(1)
    hist, t0 = [], time.time()
    for step in range(1, steps + 1):
        x, y = batches(tr, c.block_size, bs, g)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vl = sum(model(*batches(va, c.block_size, 64, g))[1].item() for _ in range(5)) / 5
            model.train()
            hist.append({"step": step, "train": loss.item(), "val": vl, "sec": time.time() - t0})
            print(f"step {step:5d} train {loss.item():.3f} val {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "chars": tok.chars, "cfg": vars(c)}, out)
    json.dump(hist, open(log, "w"), indent=1)
    model.eval()
    prompt = torch.tensor([tok.encode("please quote ")])
    print("sample:", repr(tok.decode(model.generate(prompt, 60, temperature=0)[0].tolist())))
    return model, tok


def load(path="minigpt.pt"):
    ck = torch.load(path)
    tok = CharTok("".join(ck["chars"]))
    m = MiniGPT(Config(**ck["cfg"]))
    m.load_state_dict(ck["state"]); m.eval()
    return m, tok


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    out = sys.argv[2] if len(sys.argv) > 2 else "minigpt.pt"
    log = sys.argv[3] if len(sys.argv) > 3 else "train_log.json"
    train(steps, out, log=log)
