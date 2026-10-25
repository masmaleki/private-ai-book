"""make_figures.py - charts for chapter 20.
loadtest_mock_results.json was measured against mock_llm_server.py (NOT a GPU)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "assets/img/ch20"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig); print("wrote", name)

# 1) load test curve (mock server)
r = json.loads((Path(__file__).parent / "loadtest_mock_results.json").read_text())
c = [x["conc"] for x in r]
fig, ax1 = plt.subplots(figsize=(6, 3.3))
ax1.plot(c, [x["tps"] for x in r], "o-", color="#27ae60", label="throughput tok/s")
ax1.set_xlabel("concurrent requests"); ax1.set_ylabel("output tokens/s", color="#27ae60")
ax2 = ax1.twinx()
ax2.plot(c, [x["p50"] for x in r], "s--", color="#2980b9", label="p50 latency")
ax2.plot(c, [x["p95"] for x in r], "^--", color="#c0392b", label="p95 latency")
ax2.set_ylabel("latency s")
ax1.axvline(8, color="grey", lw=0.8); ax1.text(8.4, 120, "mock capacity = 8", fontsize=8)
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.9), fontsize=8)
ax1.set_title("Mock server: throughput saturates, latency grows (not a GPU!)", fontsize=9)
save(fig, "loadtest_mock.png")

# 2) Little's law: requests in flight vs think time for 20 users
think = np.linspace(20, 300, 100)          # seconds between requests per user
fig, ax = plt.subplots(figsize=(6, 3.2))
for lat in (5, 10, 20):
    ax.plot(think, 20 * lat / (think + lat), label=f"latency {lat}s")
ax.set_xlabel("think time between requests per user (s)")
ax.set_ylabel("avg requests in flight")
ax.set_title("20 sales users, Little's law: L = arrival rate x latency", fontsize=9)
ax.grid(alpha=0.3); ax.legend()
save(fig, "littles_law.png")

# 3) prefill vs decode share of latency (formula, assumed speeds)
prompt = np.array([500, 2000, 8000, 16000])
prefill_tps, decode_tps, out_tokens = 4000, 40, 300     # assumptions
pre = prompt / prefill_tps; dec = np.full(4, out_tokens / decode_tps)
fig, ax = plt.subplots(figsize=(6, 3))
labels = [f"{p//1000 if p>=1000 else p}{'k' if p>=1000 else ''} prompt" for p in prompt]
ax.barh(labels, pre, color="#8e44ad", label="prefill (time to first token)")
ax.barh(labels, dec, left=pre, color="#e67e22", label="decode 300 tokens")
ax.set_xlabel("seconds (assumed: prefill 4000 tok/s, decode 40 tok/s)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
save(fig, "prefill_decode.png")
