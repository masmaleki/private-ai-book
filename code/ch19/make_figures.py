"""make_figures.py - charts for chapter 18 (synthetic / formula-based data only)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from quantize import quant_symmetric, quant_groupwise
from vram_calc import SCENARIOS

OUT = Path(__file__).resolve().parents[2] / "assets/img/ch18"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig); print("wrote", name)

# 1) bit layout of number formats
fmts = [("fp32", 1, 8, 23), ("bf16", 1, 8, 7), ("fp16", 1, 5, 10),
        ("fp8 E4M3", 1, 4, 3), ("int8", 0, 0, 8), ("int4", 0, 0, 4)]
fig, ax = plt.subplots(figsize=(7, 3))
for i, (n, s, e, m) in enumerate(fmts):
    y = len(fmts) - i
    ax.barh(y, s, color="#c0392b"); ax.barh(y, e, left=s, color="#2980b9")
    ax.barh(y, m, left=s + e, color="#27ae60" if s else "#8e44ad")
    ax.text(s + e + m + 0.5, y, f"{s+e+m} bits", va="center")
ax.set_yticks(range(len(fmts), 0, -1), [f[0] for f in fmts])
ax.set_xlim(0, 38); ax.set_xlabel("bits")
ax.legend(["sign", "exponent", "mantissa / integer"], loc="lower right")
save(fig, "formats.png")

# 2) weight histogram with int4 levels (per-tensor vs one group)
rng = np.random.default_rng(42)
W = rng.normal(0, 0.02, 4096 * 64)
W[rng.choice(W.size, 20, replace=False)] *= 40
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
for ax, (title, scale) in zip(axes, [
        ("int4, one scale per tensor", np.abs(W).max() / 7),
        ("int4, scale of one 32-group", np.abs(W[:32]).max() / 7)]):
    ax.hist(W, bins=200, range=(-0.1, 0.1), color="#7f8c8d")
    for k in range(-7, 8):
        if abs(k * scale) <= 0.1:
            ax.axvline(k * scale, color="#c0392b", lw=0.8)
    ax.set_title(title, fontsize=9); ax.set_xlabel("weight value")
save(fig, "levels.png")

# 3) error by method
labels, errs = [], []
for bits in (8, 4):
    q, s = quant_symmetric(W, bits)
    labels.append(f"int{bits}\nper-tensor"); errs.append(np.abs(W - q*s).mean())
for g in (128, 32):
    labels.append(f"int4\ngroup={g}")
    errs.append(np.abs(W - quant_groupwise(W, 4, g)).mean())
rel = np.array(errs) / np.abs(W).mean()
fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(labels, rel, color=["#2980b9", "#c0392b", "#27ae60", "#16a085"])
ax.bar_label(bars, fmt="%.3f"); ax.set_ylabel("relative mean abs error")
ax.set_title("Synthetic matrix with 20 outliers (lower is better)", fontsize=9)
save(fig, "error_by_method.png")

# 4) VRAM stacked bars for scenarios
names = ["A", "B", "C", "C'"]
b = [s.budget() for s in SCENARIOS]
w = [x["weights_gb"] for x in b]; kv = [x["kv_gb"] for x in b]
ov = [x["overhead_gb"] for x in b]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(names, w, label="weights", color="#2980b9")
ax.bar(names, kv, bottom=w, label="KV cache", color="#e67e22")
ax.bar(names, ov, bottom=np.add(w, kv), label="overhead", color="#95a5a6")
for x, t in enumerate(np.add(np.add(w, kv), ov)):
    ax.text(x, t + 1.5, f"{t:.0f}", ha="center")
for cap in (24, 48, 80):
    ax.axhline(cap, ls="--", lw=0.8, color="k")
    ax.text(3.45, cap + 1, f"{cap} GB", fontsize=8)
ax.set_ylabel("GiB"); ax.legend(loc="upper left")
save(fig, "vram_scenarios.png")

# 5) KV cache vs concurrent users
users = np.arange(1, 33)
fig, ax = plt.subplots(figsize=(6, 3.2))
for ctx in (4096, 8192, 16384, 32768):
    ax.plot(users, 128 * 1024 * ctx * users / 1024**3, label=f"{ctx//1024}k tokens")
ax.axhline(14.9, ls="--", color="#2980b9"); ax.text(1, 16, "bf16 weights, 8B-class")
ax.set_xlabel("concurrent sequences"); ax.set_ylabel("KV cache GiB")
ax.set_title("8B-class example config: 128 KB per token", fontsize=9)
ax.legend()
save(fig, "kv_vs_users.png")

# 6) decode ceiling vs bandwidth
bw = np.linspace(50, 3000, 100)
fig, ax = plt.subplots(figsize=(6, 3.2))
for label, gb in [("8B bf16 (16 GB)", 16), ("8B 4-bit (4.5 GB)", 4.5),
                  ("32B 4-bit (18 GB)", 18), ("70B bf16 (140 GB)", 140)]:
    ax.plot(bw, bw / gb, label=label)
ax.set_xlabel("memory bandwidth GB/s"); ax.set_ylabel("max tokens/s (1 user)")
ax.set_yscale("log"); ax.legend(fontsize=8)
ax.set_title("Theoretical ceiling = bandwidth / bytes of weights", fontsize=9)
save(fig, "decode_ceiling.png")
