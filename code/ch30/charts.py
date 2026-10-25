"""charts.py - figures for chapter 29 (synthetic data)."""
import sys; sys.path.insert(0, "../common")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fa_plot import fa, setup, save
from pricing import win_prob, optimise, model, X_of, hist

setup()
cost50, cost90, qty = 7.585, 8.4882, 2500
grid = np.round(np.arange(7.6, 10.62, 0.01), 2)
lab = {"lead_time_focused": fa("حساس به Lead Time، تحویل سریع"),
       "price_focused": fa("حساس به قیمت"), "balanced": fa("متعادل")}
col = {"lead_time_focused": "#1f3a5f", "price_focused": "#c9563c",
       "balanced": "#e0a43a"}
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for s, fast in [("lead_time_focused", 1), ("balanced", 0), ("price_focused", 0)]:
    ax.plot(grid, win_prob(s, grid / cost50, fast, 3, qty), color=col[s],
            label=lab[s], lw=2)
ax.axvline(cost50, color="gray", ls=":")
ax.text(cost50 + 0.05, 0.55, fa("هزینه P50"), fontsize=8, color="gray")
ax.set_xlabel(fa("قیمت واحد پیشنهادی (یورو)")); ax.set_ylabel(fa("احتمال برد"))
ax.set_ylim(0, 1); ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.05, 0.02))
save(fig, "ch30", "winprob_curves.png")

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=False)
for ax, (s, fast) in zip(axs, [("price_focused", 0), ("lead_time_focused", 1)]):
    r = optimise(cost50, cost90, qty, s, fast, 3)
    ep = win_prob(s, grid / cost50, fast, 3, qty) * (grid - cost50) * qty
    ax.plot(grid, ep, color=col[s], lw=2)
    ax.axvspan(grid[0], r["floor"], color="#e8a39a", alpha=0.4)
    ax.axvline(r["price"], color="k", ls="--")
    ax.set_title(lab[s], fontsize=9)
    ax.set_xlabel(fa("قیمت (یورو)"), fontsize=8)
    ax.text(r["price"], ax.get_ylim()[1] * 0.92, f" {r['price']:.2f}", fontsize=8)
axs[0].set_ylabel(fa("سود مورد انتظار (یورو)"))
axs[0].text(7.7, axs[0].get_ylim()[1] * 0.5, fa("زیر کف"), fontsize=8,
            color="#c9563c")
save(fig, "ch30", "expected_profit.png")

te = hist.iloc[3200:]
p = model.predict_proba(X_of(te))[:, 1]
b = pd.cut(p, np.linspace(0, 1, 6))
c = pd.DataFrame({"b": b, "p": p, "y": te.won.values}).groupby("b", observed=True).mean()
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.plot([0, 1], [0, 1], ls="--", color="gray")
ax.plot(c.p, c.y, "o-", color="#1f3a5f")
ax.set_xlabel(fa("احتمال برد پیش‌بینی‌شده")); ax.set_ylabel(fa("نرخ برد واقعی"))
ax.set_title(fa("کالیبراسیون مدل احتمال برد"), fontsize=10)
save(fig, "ch30", "winprob_calibration.png")

# price build-up (waterfall) for the lead-time customer, illustrative split
parts = [("هزینه‌ی مورد انتظار P50", 7.585), ("پوشش ریسک تا P90", 8.4882 - 7.585),
         ("حاشیه‌ی حداقل ۴٪", 8.83 - 8.4882), ("ارزش تحویل سریع", 10.00 - 8.83)]
fig, ax = plt.subplots(figsize=(6.0, 2.8))
bottom = 0
cols = ["#7fa7d1", "#e0a43a", "#6b9e5a", "#1f3a5f"]
for (name, v), c_ in zip(parts, cols):
    ax.bar(fa(name), v, bottom=bottom, color=c_)
    ax.text(fa(name), bottom + v / 2, f"{v:.2f}", ha="center", va="center",
            fontsize=8, color="white")
    bottom += v
ax.set_ylabel(fa("یورو در هر عدد")); ax.tick_params(axis="x", labelsize=7)
ax.set_title(fa("اجزای یک قیمت فروش نمونه (۱۰٫۰۰ یورو، ساختگی)"), fontsize=10)
save(fig, "ch30", "price_buildup.png")
