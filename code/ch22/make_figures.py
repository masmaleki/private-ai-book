"""make_figures.py - charts for chapter 21 (prompt sizes measured, curves by formula)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prompts import SYSTEM_V4, FEW_SHOT

OUT = Path(__file__).resolve().parents[2] / "assets/img/ch21"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig); print("wrote", name)

# 1) prompt anatomy: approx tokens (chars / 4) of each part of the v4 prompt
rules, schema = SYSTEM_V4.split("JSON Schema:")
email = ("Hallo Sara, wir brauchen 2.500 Stk. STM32F407VGT6 bis 15.11.2026. "
         "Der Liefertermin ist kritisch. Gruss, Markus")
parts = {"rules": len(rules) / 4, "JSON schema": len(schema) / 4,
         "few-shot": len(json.dumps(FEW_SHOT)) / 4, "email": len(email) / 4}
for k, v in parts.items():
    print(f"{k:<12} ~{v:.0f} tokens")
fig, ax = plt.subplots(figsize=(6.5, 1.8))
left = 0
for (k, v), c in zip(parts.items(), ["#2980b9", "#8e44ad", "#16a085", "#e67e22"]):
    ax.barh(0, v, left=left, color=c); ax.text(left + v / 2, 0, f"{k}\n~{v:.0f}",
                                             ha="center", va="center", color="white",
                                             fontsize=8)
    left += v
ax.set_yticks([]); ax.set_xlabel("approx. tokens (chars/4)")
save(fig, "prompt_anatomy.png")

# 2) success probability with retries: 1 - (1-p)^k
k = np.arange(1, 6)
fig, ax = plt.subplots(figsize=(6, 3.2))
for p in (0.6, 0.8, 0.9, 0.97):
    ax.plot(k, 1 - (1 - p) ** k, "o-", label=f"p(valid per try)={p}")
ax.set_xticks(k); ax.set_xlabel("max attempts"); ax.set_ylabel("P(valid within k)")
ax.set_ylim(0.55, 1.01); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("Retries help only if failures are independent (formula)", fontsize=9)
save(fig, "retry_curve.png")
