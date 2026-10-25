"""charts.py - figures for chapter 13 (synthetic data). Run embeddings.py first."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from embeddings import cosine, ngram_vec, pairs

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch13"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False})
E = np.load("embedding_table.npy")
vocab = open("vocab.txt").read().split("\n")
idx = {w: i for i, w in enumerate(vocab)}
FAM = {"tps5430": 0, "tps5450": 0, "lm2596": 0, "regulator": 0, "buck": 0, "inductor": 0,
       "converter": 0, "stm32f407": 1, "stm32f103": 1, "atmega328p": 1, "microcontroller": 1,
       "mcu": 1, "firmware": 1, "flash": 1, "grm188": 2, "cl10b104": 2, "capacitor": 2,
       "ceramic": 2, "x7r": 2, "decoupling": 2}
COL = ["#1f5f8b", "#e08a1e", "#2e8b57"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg")
    plt.close(fig)
    print("wrote", name)


# 1) PCA of the embedding table (manual, with numpy)
words = list(FAM)
X = E[[idx[w] for w in words]]
Xc = X - X.mean(0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
P = Xc @ Vt[:2].T
fig, ax = plt.subplots(figsize=(5.6, 4.0))
groups = []                       # merge labels of (almost) coincident points
for (x, y), w in zip(P, words):
    ax.scatter(x, y, color=COL[FAM[w]], s=30)
    for g in groups:
        if abs(g[0] - x) < 0.25 and abs(g[1] - y) < 0.15:
            g[2].append(w)
            break
    else:
        groups.append([x, y, [w]])
placed = []
for x, y, ws in sorted(groups, key=lambda g: -g[1]):
    k = sum(1 for px, py in placed if abs(px - x) < 0.6 and abs(py - y) < 0.4)
    placed.append((x, y))
    ax.annotate(" / ".join(ws), (x, y), xytext=(12 + 8 * k, -2 - 9 * k), textcoords="offset points",
                fontsize=7.5, ha="right" if x > 2 else "left",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5) if k else None)
for c, lab in zip(COL, ["regulator words", "MCU words", "capacitor words"]):
    ax.scatter([], [], color=c, label=lab)
ax.legend(frameon=False, fontsize=8)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("8-d co-occurrence embeddings projected to 2-d (PCA)")
save(fig, "pca_embeddings.svg")

# 2) cosine heatmap
sel = ["tps5430", "tps5450", "lm2596", "stm32f407", "stm32f103", "grm188", "cl10b104",
       "inductor", "firmware"]
M = np.array([[cosine(E[idx[a]], E[idx[b]]) for b in sel] for a in sel])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(sel)), sel, rotation=60, fontsize=8)
ax.set_yticks(range(len(sel)), sel, fontsize=8)
for i in range(len(sel)):
    for j in range(len(sel)):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(M[i, j]) > .6 else "black")
fig.colorbar(im, fraction=0.046)
ax.set_title("cosine similarity of learned vectors")
save(fig, "cosine_heatmap.svg")

# 3) one-hot vs dense
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
oh = np.eye(6)[[0, 1, 4]]
ax[0].imshow(oh, cmap="Greys", vmin=0, vmax=1)
ax[0].set_yticks(range(3), ["tps5430", "tps5450", "grm188"])
ax[0].set_xticks(range(6), [f"w{i}" for i in range(6)], fontsize=7)
ax[0].set_title("one-hot (vocab = 6)", fontsize=9)
dense = E[[idx["tps5430"], idx["tps5450"], idx["grm188"]]]
dense = dense / np.abs(dense).max()
ax[1].imshow(dense, cmap="RdBu_r", vmin=-1, vmax=1)
ax[1].set_yticks(range(3), ["tps5430", "tps5450", "grm188"])
ax[1].set_xticks(range(8), [f"d{i}" for i in range(8)], fontsize=7)
ax[1].set_title("dense embedding (dim = 8)", fontsize=9)
save(fig, "onehot_vs_dense.svg")

# 4) character n-gram similarity of MPN pairs
labels = [f"{a}\nvs {b}" for a, b in pairs]
vals = [cosine(ngram_vec(a), ngram_vec(b)) for a, b in pairs]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bars = ax.barh(range(len(vals)), vals, color="#1f5f8b")
ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.set_yticks(range(len(vals)), labels, fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("cosine of character 3-gram vectors")
ax.set_title("string similarity is not electrical compatibility")
save(fig, "mpn_ngram_similarity.svg")

# 5) cosine as angle: three 2-d vectors
fig, ax = plt.subplots(figsize=(3.8, 3.4))
vs = {"A buck 36V": (0.95, 0.30), "B buck 40V": (0.90, 0.40), "C capacitor": (0.20, 0.95)}
for (n, (x, y)), c in zip(vs.items(), COL):
    ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.text(x + 0.02, y + 0.02, n, fontsize=8, color=c)
ax.set_xlim(0, 1.3)
ax.set_ylim(0, 1.15)
ax.set_aspect("equal")
ax.set_title("small angle = high cosine", fontsize=9)
save(fig, "cosine_angle.svg")
