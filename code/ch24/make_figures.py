"""make_figures.py - figures for chapter 23 (English labels, example data)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from rag import load_docs, CHUNKERS

OUT = "../../assets/img/ch23"; os.makedirs(OUT, exist_ok=True)

# 1) chunk sizes of the fictional datasheet under three strategies
doc = [d for d in load_docs() if d.meta["doc_id"] == "DS-0001"][0]
fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), sharey=True)
for ax, (name, fn) in zip(axes, CHUNKERS.items()):
    chunks = fn(doc)
    colors = ["#c0392b" if "TA = 105" in c and "| Parameter |" not in c else
              "#4c9a2a" if "TA = 105" in c else "#7f8c8d" for c in chunks]
    ax.bar(range(1, len(chunks) + 1), [len(c.split()) for c in chunks], color=colors)
    ax.set_title(name); ax.set_xlabel("chunk #")
axes[0].set_ylabel("words")
fig.suptitle("DS-0001 chunked 3 ways (green: 105 C row with table header, "
             "red: row without header)", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/chunk_sizes.png", dpi=160); plt.close(fig)

# 2) BM25 term-frequency saturation
tf = np.arange(0, 11)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for k1 in (0.5, 1.2, 1.5, 3.0):
    ax.plot(tf, tf * (k1 + 1) / (tf + k1), marker="o", ms=3, label=f"k1={k1}")
ax.plot(tf, tf, "k--", lw=0.8, label="raw tf")
ax.set_ylim(0, 5); ax.set_xlabel("term frequency in chunk (tf)")
ax.set_ylabel("tf contribution"); ax.legend(fontsize=8)
ax.set_title("BM25 saturates: the 10th mention adds little")
fig.tight_layout(); fig.savefig(f"{OUT}/bm25_saturation.png", dpi=160); plt.close(fig)

# 3) rank of the correct chunk per query and method (from demo_search.py)
res = json.load(open("search_ranks.json"))
labels = [r["query"][:30] for r in res]
m = np.array([[r if r else 21 for r in x["ranks"]] for x in res], dtype=float)
fig, ax = plt.subplots(figsize=(7, 3.6))
im = ax.imshow(np.minimum(m, 21), cmap="RdYlGn_r", vmin=1, vmax=21, aspect="auto")
for i in range(m.shape[0]):
    for j in range(3):
        ax.text(j, i, "miss" if m[i, j] > 20 else int(m[i, j]), ha="center",
                va="center", fontsize=8)
ax.set_xticks(range(3), ["BM25", "n-gram vector", "hybrid RRF"])
ax.set_yticks(range(len(labels)), labels, fontsize=8)
ax.set_title("Rank of the correct chunk (1 is best; demo_search.py)")
fig.tight_layout(); fig.savefig(f"{OUT}/rank_heatmap.png", dpi=160); plt.close(fig)

# 4) RRF weight by rank
r = np.arange(1, 51)
fig, ax = plt.subplots(figsize=(6.5, 3.0))
for k in (10, 60):
    ax.plot(r, 1 / (k + r), label=f"k={k}")
ax.set_xlabel("rank in one result list"); ax.set_ylabel("1 / (k + rank)")
ax.set_title("Reciprocal Rank Fusion: weight decays slowly with k=60")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/rrf_weights.png", dpi=160)
print("ok")
