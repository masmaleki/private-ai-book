"""make_figures.py - charts for chapter 19 (hypothetical example scores only)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from selection_matrix import criteria, candidates

OUT = Path(__file__).resolve().parents[2] / "assets/img/ch19"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig); print("wrote", name)

# 1) radar chart of the three hypothetical candidates
keys = list(criteria)
ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist() + [0]
fig = plt.figure(figsize=(5, 4.4)); ax = plt.subplot(polar=True)
for name, sc in candidates.items():
    v = [sc[k] for k in keys] + [sc[keys[0]]]
    ax.plot(ang, v, label=name); ax.fill(ang, v, alpha=0.1)
ax.set_xticks(ang[:-1], keys, fontsize=8); ax.set_ylim(0, 5)
ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.1), fontsize=8)
save(fig, "radar.png")

# 2) sensitivity of total score to the weight of internal_bench
ws = np.linspace(0.1, 0.5, 41)
fig, ax = plt.subplots(figsize=(6, 3.2))
for name, sc in candidates.items():
    tot = []
    for w in ws:
        cw = dict(criteria); delta = w - cw["internal_bench"]
        cw["internal_bench"] = w; cw["vram_fit"] -= delta   # move weight from vram
        tot.append(sum(cw[k] * sc[k] for k in cw))
    ax.plot(ws, tot, label=name, ls="--" if "C" in name else "-")
ax.axvline(0.30, color="grey", lw=0.8); ax.text(0.305, 3.4, "chosen weight", fontsize=8)
ax.set_xlabel("weight of internal_bench (taken from vram_fit)")
ax.set_ylabel("weighted score"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("Model-C is excluded by the licence gate regardless", fontsize=9)
save(fig, "sensitivity.png")

# 3) safetensors file layout (illustration)
fig, ax = plt.subplots(figsize=(7, 1.9)); ax.axis("off")
blocks = [("8 bytes\nheader length", 1.2, "#95a5a6"),
          ("JSON header\nnames, dtype, shape, offsets", 3.0, "#2980b9"),
          ("raw tensor bytes\n(just numbers, no code)", 5.0, "#27ae60")]
x = 0
for label, w, c in blocks:
    ax.add_patch(plt.Rectangle((x, 0), w, 1, color=c, alpha=0.85))
    ax.text(x + w / 2, 0.5, label, ha="center", va="center", color="white", fontsize=9)
    x += w
ax.set_xlim(0, x); ax.set_ylim(-0.1, 1.1)
save(fig, "safetensors_layout.png")
