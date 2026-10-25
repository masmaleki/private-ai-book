"""charts.py - figures for chapter 32 (fictional data, real agent runs)."""
import sys; sys.path.insert(0, "../common")
import json, numpy as np, matplotlib.pyplot as plt
from fa_plot import setup, fa, save
import agent
setup()
res = json.load(open("eval_results.json"))
faults = [0.0, 0.05, 0.1, 0.2, 0.3]
fig, ax = plt.subplots(figsize=(8, 4))
for r, mk, lab in [(0, "o-", "بدون تلاش دوباره"), (2, "s-", "تا ۲ تلاش دوباره")]:
    ys = [res[f"{r}_{f}"]["DONE"] / 200 for f in faults]
    ax.plot([f * 100 for f in faults], ys, mk, label=fa(lab))
    for f, y in zip(faults, ys):
        ax.text(f * 100, y + 0.03, f"{y:.2f}", ha="center", fontsize=8)
ax.set_xlabel(fa("احتمال Timeout هر فراخوانی ابزار (درصد)"))
ax.set_ylabel(fa("سهم کارهای کامل‌شده"))
ax.set_ylim(0, 1.1); ax.grid(alpha=.3); ax.legend(loc="lower left")
ax.set_title(fa("۲۰۰ کار ساختگی؛ هر ابزار با احتمال مشخص Timeout می‌دهد"))
save(fig, "ch33", "success_vs_faults")

fig, ax = plt.subplots(figsize=(8, 3.6))
tm = [res[f"2_{f}"]["timeouts"] for f in faults]
ms = [res[f"2_{f}"]["mean_steps"] for f in faults]
ax.bar([f"{int(f*100)}%" for f in faults], tm, color="#e0a458")
ax.set_ylabel(fa("تعداد کل Timeoutها")); ax.set_xlabel(fa("احتمال Timeout"))
ax2 = ax.twinx(); ax2.plot(range(5), ms, "o-", color="#4a7bb7")
ax2.set_ylabel(fa("میانگین ردیف‌های Trace")); ax2.set_ylim(4, 6)
ax2.spines["right"].set_visible(True)
ax.set_title(fa("تلاش دوباره موفقیت می‌خرد، به قیمت گام و زمان بیشتر"))
save(fig, "ch33", "retry_cost")

task = {"customer": "AlpenBoard Example GmbH", "mpn": "STM32F407VGT6", "qty": 2500}
runs = [("اجرای عادی (dry-run)", agent.run(task)),
        ("قیمت زیر کف", agent.run(task | {"force_price": 8.50})),
        ("دستور تزریق‌شده", agent.run(task, obey_injection=True)),
        ("حلقه‌ی بی‌پایان", agent.run(task, loop_bug=True)),
        ("کلید توقف", agent.run(task, policy=agent.Policy(kill_switch=True)))]
col = {"lookup_part": "#9db8d6", "predict_price": "#4a7bb7", "find_alternates": "#7fb27f",
       "create_quote_draft": "#e0a458"}
fig, ax = plt.subplots(figsize=(9, 3.8))
for i, (lab, r) in enumerate(runs):
    for t in r["trace"]:
        x = t["step"] - 1
        if "blocked" in t:
            ax.barh(i, 0.9, left=x, color="#d6806b"); ax.text(x + 0.45, i, "X", ha="center", va="center", color="white", weight="bold")
        elif "final" in t:
            ax.barh(i, 0.9, left=x, color="#555"); ax.text(x + 0.45, i, "OK", ha="center", va="center", color="white", fontsize=8)
        else:
            ax.barh(i, 0.9, left=x, color=col[t["tool"]])
ax.set_yticks(range(len(runs))); ax.set_yticklabels([fa(l) for l, _ in runs])
ax.invert_yaxis(); ax.set_xlabel(fa("گام")); ax.set_xticks(np.arange(8) + 0.45); ax.set_xticklabels(range(1, 9))
from matplotlib.patches import Patch
h = [Patch(color=c, label=n) for n, c in col.items()] + [Patch(color="#d6806b", label="blocked"), Patch(color="#555", label="final")]
ax.legend(handles=h, fontsize=7, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.35))
fig.tight_layout(); save(fig, "ch33", "trace_runs")
