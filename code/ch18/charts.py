"""charts.py - figures for chapter 17 (toy numbers and arithmetic only)."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sampling import TOKENS, LOGITS, softmax, with_temperature, top_k, top_p, kv_cache_bytes

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch17"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, format="svg"); plt.close(fig); print("wrote", name)


# 1) temperature
fig, ax = plt.subplots(figsize=(6.0, 2.8))
w = 0.2
for i, T in enumerate([0.5, 1.0, 2.0]):
    pr = with_temperature(LOGITS, T)
    b = ax.bar(np.arange(4) + (i - 1) * w, pr, w, label=f"T = {T}")
ax.set_xticks(range(4), [f'"{t}"' for t in TOKENS])
ax.set_ylabel("probability"); ax.set_xlabel('next token after "lead time "')
ax.legend(frameon=False); ax.set_title("temperature reshapes the same logits", fontsize=9)
save(fig, "temperature.svg")

# 2) top-k / top-p
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
for ax, (name, pr) in zip(axs, [("full softmax", softmax(LOGITS)), ("top-k, k = 2", top_k(LOGITS, 2)),
                               ("top-p, p = 0.9", top_p(LOGITS, 0.9))]):
    ax.bar(range(4), pr, color=["#1f5f8b" if v > 0 else "#cccccc" for v in pr])
    for i, v in enumerate(pr):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(4), [f'"{t}"' for t in TOKENS], fontsize=8); ax.set_title(name, fontsize=9)
axs[0].set_ylabel("probability")
save(fig, "topk_topp.svg")

# 3) KV cache vs context length (7B-class, fp16)
ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768])
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for kvh, lab in [(32, "MHA (32 KV heads)"), (8, "GQA (8 KV heads)")]:
    ax.plot(ctx, [kv_cache_bytes(32, kvh, 128, t) / 2**30 for t in ctx], "o-", label=lab)
ax.axhline(13.5e9 / 2**30, color="0.5", ls="--", lw=1)
ax.text(1100, 13.5e9 / 2**30 + 0.6, "fp16 weights of a 6.74 B model", fontsize=7, color="0.4")
ax.set_xscale("log", base=2); ax.set_xlabel("context length (tokens), one sequence")
ax.set_ylabel("KV cache (GiB)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("32 layers, head_dim 128, fp16", fontsize=9)
save(fig, "kv_cache.svg")

# 4) prefill vs decode timeline (schematic)
fig, ax = plt.subplots(figsize=(6.6, 2.0))
ax.broken_barh([(0, 3)], (0.6, 0.8), color="#e08a1e")
ax.text(1.5, 1.0, "prefill: whole prompt\n(parallel, compute-bound)", ha="center", va="center", fontsize=7, color="white")
for i in range(10):
    ax.broken_barh([(3.2 + i * 0.9, 0.75)], (0.6, 0.8), color="#1f5f8b")
    ax.text(3.2 + i * 0.9 + 0.37, 1.0, f"t{i + 1}", ha="center", va="center", fontsize=7, color="white")
ax.annotate("time to first token", xy=(3.1, 0.55), xytext=(3.1, 0.1), ha="center", fontsize=7,
            arrowprops=dict(arrowstyle="->"))
ax.text(8.0, 1.75, "decode: one token per step (sequential, memory-bandwidth-bound)", ha="center", fontsize=7)
ax.set_xlim(0, 12.5); ax.set_ylim(0, 2.1); ax.set_yticks([]); ax.set_xlabel("time (schematic)")
for s in ("left",):
    ax.spines[s].set_visible(False)
save(fig, "prefill_decode.svg")

# 5) static vs continuous batching (schematic)
fig, axs = plt.subplots(2, 1, figsize=(6.6, 3.0), sharex=True)
lens = [3, 8, 5, 2]
cols = ["#1f5f8b", "#e08a1e", "#2e8b57", "#8b3a62"]
# static: all wait for the longest
for r, (L, c) in enumerate(zip(lens, cols)):
    axs[0].broken_barh([(0, L)], (r, 0.8), color=c)
    axs[0].broken_barh([(L, 8 - L)], (r, 0.8), color="#eeeeee")
axs[0].broken_barh([(8, 4)], (0, 0.8), color="#5d7fa3"); axs[0].text(10, 0.4, "new req.", ha="center", va="center", fontsize=7, color="white")
axs[0].set_title("static batching: slots idle until the longest request ends", fontsize=8)
# continuous: new requests join as soon as a slot frees up
for r, (L, c) in enumerate(zip(lens, cols)):
    axs[1].broken_barh([(0, L)], (r, 0.8), color=c)
axs[1].broken_barh([(3, 4)], (0, 0.8), color="#5d7fa3"); axs[1].text(5, 0.4, "new req.", ha="center", va="center", fontsize=7, color="white")
axs[1].broken_barh([(2, 5)], (3, 0.8), color="#a0522d"); axs[1].text(4.5, 3.4, "new req.", ha="center", va="center", fontsize=7, color="white")
axs[1].set_title("continuous batching: a freed slot is refilled at the next step", fontsize=8)
for ax in axs:
    ax.set_yticks([0.4, 1.4, 2.4, 3.4], ["slot 1", "slot 2", "slot 3", "slot 4"], fontsize=7)
axs[1].set_xlabel("decode steps (schematic)")
save(fig, "batching.svg")

# 6) invented part numbers vs temperature (real runs of sweep.py, 100 samples each)
import json, re
sw = json.load(open("sweep.json"))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.plot([r["T"] for r in sw], [r["plain"] for r in sw], "o-", label="temperature only")
ax.plot([r["T"] for r in sw], [r["top_p_0.9"] for r in sw], "s--", label="temperature + top-p 0.9")
ax.set_xlabel("temperature T"); ax.set_ylabel("invented MPNs per 100 samples")
ax.legend(frameon=False, fontsize=8); ax.set_ylim(-3, 100)
ax.set_title('mini-GPT, prompt "rfq: ", synthetic data', fontsize=9)
save(fig, "invented_mpn.svg")

# 7) naive re-computation vs KV cache (real timings from kvcache.out, CPU)
rows = [tuple(map(float, re.findall(r"[\d.]+", l)))
        for l in open("kvcache.out") if l.startswith("ctx")]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="no cache: recompute prefix")
ax.plot([r[0] for r in rows], [r[2] for r in rows], "s-", label="with KV cache: one new token")
ax.set_xlabel("tokens already in context"); ax.set_ylabel("ms per generated token")
ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=8)
ax.set_title("mini-GPT on CPU (your timings will differ)", fontsize=9)
save(fig, "kv_timing.svg")
