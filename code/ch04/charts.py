"""charts.py - Chapter 4 figures (assumed, illustrative numbers)."""
import matplotlib.pyplot as plt
from fa_plot import fa, save

# Latency budget for request (a) "explain this price" - assumed targets, ms
steps = [("احراز هویت و سیاست", 20), ("تشخیص نیت با LLM", 400),
         ("ابزار: Offerها", 80), ("مدل قیمت", 30),
         ("نوشتن توضیح با LLM", 2500), ("ثبت Audit", 10)]
colors = ["#7f9fbf", "#c0504d", "#1f4e79", "#1f4e79", "#c0504d", "#7f9fbf"]
fig, ax = plt.subplots(figsize=(8, 2.6))
left = 0
for (name, ms), c in zip(steps, colors):
    ax.barh([0], [ms], left=left, color=c, edgecolor="white")
    if ms > 300:
        ax.text(left + ms / 2, 0, fa(f"{name}\n{ms} ms", digits=False),
                ha="center", va="center", color="white", fontsize=8)
    left += ms
ax.set_yticks([])
ax.set_xlabel(fa("میلی‌ثانیه (بودجه‌ی فرضی)"))
ax.set_title(fa("بودجه‌ی زمانی درخواست «این قیمت را توضیح بده»: سهم LLM غالب است"), pad=22)
ax.text(left, 0.45, fa(f"جمع ≈ {left/1000:.1f} ثانیه"), ha="right")
save(fig, "ch04", "latency_budget.png")

# Who calls what: share of tool calls per block for 3 walkthroughs
walks = ["توضیح قیمت", "پاک‌سازی BOM", "Tenant دفاعی"]
blocks = {"LLM خصوصی": [2, 2, 2], "جست‌وجو و RAG": [0, 1, 1],
          "مدل‌های پیش‌بینی": [1, 0, 0], "لایه‌ی ابزار امن": [2, 2, 1]}
cols = ["#c0504d", "#e3a33b", "#1f4e79", "#4f8a5b"]
fig, ax = plt.subplots(figsize=(7, 3.2))
bottom = [0, 0, 0]
for (name, vals), c in zip(blocks.items(), cols):
    ax.bar([fa(w, digits=False) for w in walks], vals, bottom=bottom,
           color=c, label=fa(name, digits=False))
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_ylabel(fa("تعداد فراخوانی در هر درخواست"))
ax.set_title(fa("کدام بلوک در هر سناریو کار می‌کند؟ (نمونه)"))
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.set_ylim(0, 7)
save(fig, "ch04", "blocks_per_walkthrough.png")
