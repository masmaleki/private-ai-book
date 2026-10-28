"""charts.py - figures for chapter 14 (toy / synthetic numbers)."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from attention import attention, softmax, Q, K, V, tokens

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch14"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg")
    plt.close(fig)
    print("wrote", name)


def heat(ax, M, xl, yl, title, fmt="{:.2f}"):
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(xl)), xl, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yl)), yl, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > .55 else "black")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("key (looked at)", fontsize=8)
    ax.set_ylabel("query (looking)", fontsize=8)


# 1) hand example weights, full vs causal
fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0))
heat(ax[0], attention(Q, K, V, False)[1], tokens, tokens, "no mask")
heat(ax[1], attention(Q, K, V, True)[1], tokens, tokens, "causal mask")
save(fig, "hand_weights.svg")

# 2) causal mask matrix for 6 tokens
toks = ["need", "500", "pcs", "STM32", "by", "May"]
n = len(toks)
M = np.tril(np.ones((n, n)))
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(M, cmap="Greens", vmin=0, vmax=1.4)
ax.set_xticks(range(n), toks, rotation=45, fontsize=8)
ax.set_yticks(range(n), toks, fontsize=8)
for i in range(n):
    for j in range(n):
        ax.text(j, i, "ok" if M[i, j] else "-inf", ha="center", va="center", fontsize=7)
ax.set_title("causal mask: row i may see columns <= i", fontsize=9)
save(fig, "causal_mask.svg")

# 3) why divide by sqrt(d): std of q.k grows with d; softmax saturates
rng = np.random.default_rng(0)
ds = [2, 8, 32, 128, 512]
raw, scaled, maxw_raw, maxw_scaled = [], [], [], []
for d in ds:
    q = rng.normal(size=(2000, d)); k = rng.normal(size=(2000, 16, d))
    s = np.einsum("nd,nkd->nk", q, k)
    raw.append(s.std()); scaled.append((s / np.sqrt(d)).std())
    maxw_raw.append(softmax(s).max(1).mean()); maxw_scaled.append(softmax(s / np.sqrt(d)).max(1).mean())
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
ax[0].plot(ds, raw, "o-", label="q.k"); ax[0].plot(ds, scaled, "s-", label="q.k / sqrt(d)")
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("d"); ax[0].set_ylabel("std of score")
ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(ds, maxw_raw, "o-", label="unscaled"); ax[1].plot(ds, maxw_scaled, "s-", label="scaled")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("d"); ax[1].set_ylabel("mean max weight (16 keys)")
ax[1].set_ylim(0, 1.05); ax[1].legend(frameon=False, fontsize=8)
save(fig, "sqrt_d_scaling.svg")
print("std raw", np.round(raw, 2), "scaled", np.round(scaled, 2))
print("maxw raw", np.round(maxw_raw, 3), "scaled", np.round(maxw_scaled, 3))

# 4) O(n^2): size of one attention-score matrix per head, fp16
ns = np.array([1_000, 4_000, 8_000, 32_000, 128_000])
mib = ns.astype(float) ** 2 * 2 / 2**20
fig, ax = plt.subplots(figsize=(5.4, 2.8))
ax.plot(ns, mib, "o-")
for x, y in zip(ns, mib):
    ax.annotate(f"{y:,.0f} MiB", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("sequence length n (tokens)")
ax.set_ylabel("n x n scores, fp16 (MiB)")
ax.set_title("one head, one layer, if the matrix were materialised", fontsize=9)
save(fig, "quadratic_memory.svg")
print("MiB", np.round(mib, 1))

# 5) illustrative attention on an RFQ sentence (hand-designed weights, not from a trained model)
sent = ["Please", "quote", "500", "pcs", "STM32F407", "and", "200", "TPS5430"]
W = np.full((len(sent), len(sent)), 0.02)
# causal: "500" (pos 2) cannot see "STM32F407" (pos 4); the part looks BACK at its quantity
W[3, 2] = 0.5; W[4, 2] = 0.5; W[4, 1] = 0.3; W[7, 6] = 0.6; W[7, 1] = 0.2
W = np.tril(W)                      # zero everything above the diagonal
W = W / W.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.0, 4.2))
heat(ax, W, sent, sent, "illustrative causal attention (hand-made)", fmt="{:.1f}")
save(fig, "rfq_attention_illustration.svg")

# 6) MHA vs GQA diagram (boxes and lines)
from matplotlib.patches import FancyBboxPatch
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, title, groups in [(axs[0], "MHA: every query head has its own K/V", [0, 1, 2, 3]),
                          (axs[1], "GQA: 2 query heads share one K/V", [0, 0, 1, 1])]:
    ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.axis("off"); ax.set_title(title, fontsize=9)
    n_kv = max(groups) + 1
    kv_x = [0.5 + i * (3.0 / max(n_kv - 1, 1)) if n_kv > 1 else 2 for i in range(n_kv)]
    if n_kv == 2:
        kv_x = [1.0, 3.0]
    for i, g in enumerate(groups):
        qx = 0.5 + i
        ax.add_patch(FancyBboxPatch((qx - 0.35, 2.1), 0.7, 0.5, boxstyle="round,pad=0.02", fc="#dbe8f4", ec="#1f5f8b"))
        ax.text(qx, 2.35, f"Q{i + 1}", ha="center", va="center", fontsize=9)
        ax.plot([qx, kv_x[g]], [2.1, 0.9], color="0.5", lw=1)
    for j, x in enumerate(kv_x):
        ax.add_patch(FancyBboxPatch((x - 0.4, 0.4), 0.8, 0.5, boxstyle="round,pad=0.02", fc="#fbe3c4", ec="#e08a1e"))
        ax.text(x, 0.65, f"K{j + 1} V{j + 1}", ha="center", va="center", fontsize=8)
    ax.text(2, 0.05, "KV cache size: 1/2 of MHA" if n_kv == 2 else "KV cache size: 1x", ha="center", fontsize=8)
save(fig, "mha_vs_gqa.svg")
