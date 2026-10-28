"""charts.py - figures for chapter 15. Run minigpt.py 600 first (writes train_log.json)."""
import json, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from param_count import count
from minigpt import rope

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch15"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, format="svg"); plt.close(fig); print("wrote", name)


# 1) where the parameters live
small = count(512, 8, 32_000)
big = count(4096, 32, 32_000, mlp_hidden=11_008, mlp_mats=3, tied=False)
cats = ["embedding", "attention", "MLP", "output head", "norms"]
def shares(c, L):
    v = [c["emb"], L * c["attn"], L * c["mlp"], c["head"], L * c["norms"] + c["final_norm"]]
    return np.array(v) / c["total"] * 100
S = np.vstack([shares(small, 8), shares(big, 32)])
fig, ax = plt.subplots(figsize=(6.4, 2.4))
left = np.zeros(2)
cols = ["#7aa6c2", "#1f5f8b", "#e08a1e", "#b5cfe0", "#888888"]
for j, (cat, col) in enumerate(zip(cats, cols)):
    ax.barh([0, 1], S[:, j], left=left, color=col, label=cat)
    for i in range(2):
        if S[i, j] > 4:
            ax.text(left[i] + S[i, j] / 2, i, f"{S[i, j]:.0f}%", ha="center", va="center", fontsize=8, color="white")
    left += S[:, j]
ax.set_yticks([0, 1], [f"d=512, L=8, tied\n{small['total'] / 1e6:.1f} M", f"7B-class config\n{big['total'] / 1e9:.2f} B"])
ax.set_xlabel("share of parameters (%)"); ax.set_xlim(0, 100)
ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28))
save(fig, "param_shares.svg")

# 2) training curve of the 600-step run
h = json.load(open("train_log.json"))
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.plot([r["step"] for r in h], [r["train"] for r in h], "o-", label="train")
ax.plot([r["step"] for r in h], [r["val"] for r in h], "s--", label="validation")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("cross-entropy (nats/char)")
ax.set_title("mini-GPT on synthetic RFQ sentences (CPU)", fontsize=9); ax.legend(frameon=False)
save(fig, "minigpt_loss.svg")

# 3) RoPE: score q.k depends on the distance only -> constant along diagonals
torch.manual_seed(0)
T, D = 24, 16
q = torch.randn(D); k = torch.randn(D)
Qs = rope(q.expand(1, 1, T, D).clone())[0, 0]
Ks = rope(k.expand(1, 1, T, D).clone())[0, 0]
M = (Qs @ Ks.T).numpy()
M = np.tril(M)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(np.where(np.tril(np.ones((T, T))) > 0, M, np.nan), cmap="RdBu_r")
ax.set_xlabel("key position"); ax.set_ylabel("query position")
ax.set_title("same q and k at every position, after RoPE", fontsize=9)
fig.colorbar(im, fraction=0.046)
save(fig, "rope_diagonals.svg")

# 4) residual stream: parameter count per layer vs d (12 d^2 rule)
ds = np.array([256, 512, 1024, 2048, 4096, 8192])
fig, ax = plt.subplots(figsize=(5.0, 2.8))
ax.plot(ds, 12 * ds.astype(float) ** 2 / 1e6, "o-")
for d in ds:
    ax.annotate(f"{12 * d * d / 1e6:,.0f}M", (d, 12 * d * d / 1e6), xytext=(4, -10), textcoords="offset points", fontsize=7)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("d_model"); ax.set_ylabel("params per layer (millions)")
ax.set_title("~12 d^2 per block (4d^2 attention + 8d^2 MLP)", fontsize=9)
save(fig, "params_per_layer.svg")
