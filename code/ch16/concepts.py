"""concepts.py - small numeric checks for chapter 15: RMSNorm vs LayerNorm, RoPE relative property,
sinusoidal positions, and a toy MoE router. Toy numbers only."""
import numpy as np
import torch
from minigpt import rope

np.set_printoptions(precision=4, suppress=True)

# RMSNorm vs LayerNorm on one vector
x = np.array([2.0, -1.0, 0.5, 4.5])
mean, var = x.mean(), x.var()
ln = (x - mean) / np.sqrt(var + 1e-6)
rms = np.sqrt((x ** 2).mean())
rn = x / rms
print("x =", x, " mean =", mean, " var =", var, " rms =", round(rms, 4))
print("LayerNorm =", ln)
print("RMSNorm   =", rn)

# sinusoidal position encoding (first 4 dims, d=8)
def sinusoidal(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((T, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    return pe
print("sinusoidal PE (d=8), positions 0..2:\n", sinusoidal(3, 8))

# RoPE: q.k after rotation depends only on the distance between positions
torch.manual_seed(0)
q = torch.randn(1, 1, 1, 8); k = torch.randn(1, 1, 1, 8)
def rotated_at(v, pos):
    seq = torch.zeros(1, 1, pos + 1, 8); seq[0, 0, pos] = v[0, 0, 0]
    return rope(seq)[0, 0, pos]
for (pq, pk) in [(5, 3), (12, 10), (40, 38), (5, 1)]:
    print(f"RoPE q@pos{pq} . k@pos{pk} (distance {pq - pk}) = {float(rotated_at(q, pq) @ rotated_at(k, pk)):.4f}")

# toy MoE router: 4 experts, top-2
rng = np.random.default_rng(3)
h = rng.normal(size=4)                  # one token's hidden state (d=4)
Wr = rng.normal(size=(4, 4))            # router weights: d -> n_experts
logits = h @ Wr
top2 = np.argsort(logits)[-2:][::-1]
w = np.exp(logits[top2] - logits[top2].max()); w /= w.sum()
print("router logits =", logits, " top-2 experts =", top2, " weights =", w)
