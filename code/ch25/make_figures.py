"""make_figures.py - figures for chapter 24 (English labels, example data)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT = "../../assets/img/ch24"; os.makedirs(OUT, exist_ok=True)

# 1) evaluation results (real output of eval_retrieval.py)
res = json.load(open("eval_results.json"))
setups, metrics = list(res), ["Recall@5", "MRR", "nDCG@5"]
x = np.arange(len(setups)); w = 0.26
fig, ax = plt.subplots(figsize=(7, 3.4))
for j, m in enumerate(metrics):
    vals = [res[s][j] for s in setups]
    ax.bar(x + (j - 1) * w, vals, w, label=m)
    for xi, v in zip(x + (j - 1) * w, vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(x, setups); ax.set_ylim(0.6, 1.05); ax.legend(fontsize=8, ncol=3)
ax.set_title("Retrieval quality on 15 example queries (eval_retrieval.py)")
fig.tight_layout(); fig.savefig(f"{OUT}/eval_bars.png", dpi=160); plt.close(fig)

# 2) nDCG position discount
i = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.bar(i, 1 / np.log2(i + 1), color="#3b7dd8")
for xi, v in zip(i, 1 / np.log2(i + 1)):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
ax.set_xticks(i); ax.set_xlabel("rank position"); ax.set_ylabel("discount 1/log2(i+1)")
ax.set_title("nDCG: a relevant chunk at rank 1 counts ~3x more than at rank 10")
fig.tight_layout(); fig.savefig(f"{OUT}/ndcg_discount.png", dpi=160); plt.close(fig)

# 3) post-filter vs pre-filter: visible results left from a top-10 list
share = np.linspace(0.01, 1, 100)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(share * 100, 10 * share, label="filter AFTER top-10 (expected)")
ax.plot(share * 100, np.full_like(share, 10), label="filter BEFORE ranking")
ax.set_xlabel("share of the index this user may see (%)")
ax.set_ylabel("results shown (of 10)"); ax.legend(fontsize=8)
ax.set_title("Post-filtering starves small tenants (illustrative)")
fig.tight_layout(); fig.savefig(f"{OUT}/prefilter.png", dpi=160); plt.close(fig)
print("ok")
