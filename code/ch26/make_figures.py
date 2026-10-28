"""make_figures.py - figures for chapter 25 (English labels, example data)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from bom_clean import clean

OUT = "../../assets/img/ch25"; os.makedirs(OUT, exist_ok=True)

# 1) Levenshtein DP table for BC847 -> BC874
a, b = "BC847", "BC874"
d = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
d[:, 0] = range(len(a) + 1); d[0, :] = range(len(b) + 1)
for i in range(1, len(a) + 1):
    for j in range(1, len(b) + 1):
        d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (a[i-1] != b[j-1]))
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.imshow(d, cmap="Blues", vmin=0, vmax=6)
for i in range(d.shape[0]):
    for j in range(d.shape[1]):
        ax.text(j, i, d[i, j], ha="center", va="center",
                color="white" if d[i, j] > 3 else "black", fontsize=11)
ax.set_xticks(range(len(b) + 1), [""] + list(b))
ax.set_yticks(range(len(a) + 1), [""] + list(a))
ax.add_patch(plt.Rectangle((len(b) - 0.5, len(a) - 0.5), 1, 1, fill=False,
                           ec="red", lw=2))
ax.set_title("Levenshtein table: BC847 -> BC874 = 2")
fig.tight_layout(); fig.savefig(f"{OUT}/lev_table.png", dpi=160); plt.close(fig)

# 2) threshold trade-off (threshold_eval.py, synthetic corruptions)
c = json.load(open("threshold_curve.json"))
t = [r["t"] for r in c]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(t, [r["auto_rate"] * 100 for r in c], label="lines auto-accepted (%)")
ax.plot(t, [r["auto_precision"] * 100 for r in c], label="correct among auto (%)")
ax.axvline(0.90, ls="--", color="gray"); ax.text(0.905, 40, "AUTO=0.90", fontsize=8)
ax.set_xlabel("confidence threshold"); ax.set_ylim(0, 105); ax.legend(fontsize=8)
ax.set_title("Auto-accept rate vs. correctness (460 synthetic inputs)")
fig.tight_layout(); fig.savefig(f"{OUT}/threshold_tradeoff.png", dpi=160)
plt.close(fig)

# 3) the 15-line example BOM: confidence and final status
lines, *_ = clean("messy_bom.csv")
colors = {"auto": "#4c9a2a", "review": "#e0a100", "unmatched": "#c0392b"}
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar([l["line"] for l in lines], [l["conf"] for l in lines],
       color=[colors[l["status"]] for l in lines])
ax.axhline(0.90, ls="--", color="gray", lw=0.8)
ax.axhline(0.60, ls=":", color="gray", lw=0.8)
for s, col in colors.items():
    ax.bar(0, 0, color=col, label=s)
ax.set_xticks(range(1, 16)); ax.set_xlabel("BOM line"); ax.set_ylabel("confidence")
ax.set_title("Example BOM: confidence per line, colour = final status")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/bom_confidence.png", dpi=160)
print("ok")
