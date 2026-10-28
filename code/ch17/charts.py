"""charts.py - figures for chapter 16 from the JSON logs written by experiments.py / sft.py.
Synthetic data only. Usage: python3 charts.py"""
import json, math, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from costs import train_flops, gpu_hours

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch16"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False})
FLOOR = 0.1347          # entropy floor printed by experiments.py


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, format="svg"); plt.close(fig); print("wrote", name)


# 1) long pretraining run with the entropy floor
h = json.load(open("long.json"))
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot([r["step"] for r in h], [r["train"] for r in h], label="train loss")
ax.plot([r["step"] for r in h], [r["val"] for r in h], "--", label="validation loss")
ax.axhline(FLOOR, color="0.4", lw=1, ls=":")
ax.text(1500, FLOOR * 0.8, f"entropy floor of the generator = {FLOOR}", fontsize=7, color="0.3")
ax.axhline(math.log(57), color="0.6", lw=1, ls=":")
ax.text(1500, math.log(57) * 1.1, "ln(57) = 4.04: random guessing", fontsize=7, color="0.3")
ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("loss (nats/char, log)")
ax.legend(frameon=False, fontsize=8); ax.set_title("mini-GPT, 3000 steps, synthetic RFQ text", fontsize=9)
save(fig, "loss_long.svg")

# 2) toy scaling: bigger model -> lower loss, with diminishing returns
s = json.load(open("scale.json"))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.plot([r["params"] for r in s], [r["val"] for r in s], "o-")
for r in s:
    ax.annotate(f"d={r['d']}\n{r['val']:.3f}", (r["params"], r["val"]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7)
ax.axhline(FLOOR, color="0.4", lw=1, ls=":"); ax.text(14000, FLOOR + 0.004, "floor", fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("parameters (log)"); ax.set_ylabel("validation loss @ 800 steps")
ax.set_ylim(0.12, 0.29); ax.set_title("same data, same steps, 4 model sizes", fontsize=9)
save(fig, "scaling_toy.svg")

# 3) memorisation: 40 sentences only
m = json.load(open("memo.json"))
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot([r["step"] for r in m], [r["train"] for r in m], label="train (40 sentences)")
ax.plot([r["step"] for r in m], [r["val"] for r in m], "--", label="validation (600 new sentences)")
ax.axhline(FLOOR, color="0.4", lw=1, ls=":")
ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("loss (log)")
ax.legend(frameon=False, fontsize=8); ax.set_title("too little data: the model memorises", fontsize=9)
save(fig, "memorization.svg")

# 4) SFT: answer loss and exact-match accuracy
sft = json.load(open("sft.json"))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
axs[0].plot([r["step"] for r in sft], [r["loss"] for r in sft], "o-")
axs[0].set_xlabel("SFT step"); axs[0].set_ylabel("loss on answer chars"); axs[0].set_yscale("log")
axs[0].set_title("SFT: 300 steps, 300 pairs", fontsize=9)
axs[1].bar(["base", "after SFT"], [0, 67], color=["#cccccc", "#1f5f8b"])
for i, v in enumerate([0, 67]):
    axs[1].text(i, v + 2, f"{v}/100", ha="center", fontsize=8)
axs[1].set_ylim(0, 100); axs[1].set_ylabel("exact-match extractions")
axs[1].set_title("100 unseen test RFQs", fontsize=9)
save(fig, "sft_result.svg")

# 5) compute: GPU-hours for several training jobs (stated assumptions)
jobs = [("LoRA SFT 7B\n20M tokens", 7e9, 20e6, 2 / 3), ("continued PT 7B\n2B tokens", 7e9, 2e9, 1),
        ("pretrain 7B\n2T tokens", 7e9, 2e12, 1), ("pretrain 7B\n15T tokens", 7e9, 15e12, 1),
        ("pretrain 70B\n15T tokens", 70e9, 15e12, 1)]
vals = [gpu_hours(train_flops(n, d)) * f for _, n, d, f in jobs]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.bar(range(len(jobs)), vals, color=["#2e8b57", "#2e8b57", "#e08a1e", "#e08a1e", "#8b3a62"])
for i, v in enumerate(vals):
    ax.text(i, v * 1.3, f"{v:,.0f}" if v >= 1 else f"{v:.1f}", ha="center", fontsize=7)
ax.set_yscale("log"); ax.set_xticks(range(len(jobs)), [j[0] for j in jobs], fontsize=7)
ax.set_ylabel("GPU-hours (log)")
ax.set_title("C = 6·N·D, 1e15 FLOP/s peak, 40% utilisation (assumptions)", fontsize=8)
save(fig, "compute_cost.svg")
