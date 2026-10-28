"""charts.py - figures for chapter 27 (synthetic data)."""
import sys; sys.path.insert(0, "../common")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fa_plot import fa, setup, save

setup()
offers = pd.read_csv("demo_offers.csv", parse_dates=["observed_at"])
test = pd.read_csv("test_predictions.csv", parse_dates=["target_date"])
met = json.load(open("metrics.json"))

g = offers[offers.part_id == 0]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for q, col in [(100, "#7fa7d1"), (2500, "#1f3a5f"), (10000, "#e0a43a")]:
    h = g[g.qty == q]
    ax.scatter(h.observed_at, h.unit_price, s=10, color=col, label=f"qty {q}")
ax.axvline(pd.Timestamp("2026-03-01"), color="#c9563c", ls="--")
ax.text(pd.Timestamp("2026-02-20"), ax.get_ylim()[0] * 1.02,
        fa("مرز آموزش/آزمون"), color="#c9563c", fontsize=8, ha="right")
ax.set_ylabel(fa("قیمت واحد (یورو)"))
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title(fa("تاریخچه‌ی پیشنهادهای STM32F407VGT6 (ساختگی)"), fontsize=10)
save(fig, "ch28", "history_part0.png")

fig, ax = plt.subplots(figsize=(5.0, 2.6))
names = [fa("آخرین قیمت"), fa("میانه‌ی ۹۰ روزه"), "LightGBM P50"]
vals = [met["b_last"]["WAPE"], met["b_med90"]["WAPE"], met["p50"]["WAPE"]]
bars = ax.bar(names, vals, color=["#7fa7d1", "#7fa7d1", "#1f3a5f"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
            ha="center", fontsize=9)
ax.set_ylabel("WAPE"); ax.set_ylim(0, max(vals) * 1.2)
ax.set_title(fa("خطای دوره‌ی آزمون؛ کمتر بهتر است"), fontsize=10)
save(fig, "ch28", "baseline_vs_model.png")

t = test[test.part_id == 0].sort_values("target_date")
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.vlines(t.target_date, t.p10, t.p90, color="#7fa7d1", lw=3, alpha=0.7,
          label=fa("بازه‌ی P10 تا P90"))
ax.scatter(t.target_date, t.p50, marker="_", s=80, color="#1f3a5f",
           label="P50", zorder=3)
ax.scatter(t.target_date, t.y_price, s=12, color="#c9563c",
           label=fa("قیمت واقعی"), zorder=4)
ax.set_ylabel(fa("قیمت واحد (یورو)"))
ax.legend(fontsize=8, frameon=False, ncol=3)
ax.set_title(fa("پیش‌بینی ۳۰ روز جلوتر در دوره‌ی آزمون ، همه‌ی تعدادها (هر ستون یک پیشنهاد)"),
             fontsize=10)
save(fig, "ch28", "band_part0.png")

imp = {"log_qty_vs_hist": .512, "log_qty": .130, "trend_30_90": .091,
       "lead_med90": .078, "lead_trend": .046, "lead_med30": .044,
       "n_offers_90": .039, "category": .027, "days_since_last": .025,
       "lifecycle": .008}   # copied from train_price.py output
fig, ax = plt.subplots(figsize=(5.4, 3.0))
k = list(imp)[::-1]
ax.barh(k, [imp[x] for x in k], color="#3d6a99")
ax.set_xlabel(fa("سهم از Gain"))
save(fig, "ch28", "importance.png")

fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.bar([fa("خام"), fa("پس از کالیبراسیون")], [0.724, 0.794],
       color=["#e0a43a", "#6b9e5a"])
ax.axhline(0.80, ls="--", color="#c9563c")
ax.text(1.45, 0.81, fa("هدف ۰٫۸۰"), ha="right", fontsize=9, color="#c9563c")
ax.set_ylim(0.5, 0.9); ax.set_ylabel(fa("پوشش P10 تا P90"))
save(fig, "ch28", "coverage.png")

seg = test.copy()
seg["e_b"] = np.abs(seg.y_price - seg.b_med90) / seg.y_price
seg["e_m"] = np.abs(seg.y_price - seg.p50) / seg.y_price
s = seg.groupby("category")[["e_b", "e_m"]].mean()
fig, ax = plt.subplots(figsize=(5.6, 2.8))
x = np.arange(len(s))
ax.bar(x - 0.2, s.e_b, 0.4, color="#7fa7d1", label=fa("میانه‌ی ۹۰ روزه"))
ax.bar(x + 0.2, s.e_m, 0.4, color="#1f3a5f", label="LightGBM")
ax.set_xticks(x, s.index); ax.set_ylabel("MAPE")
ax.legend(fontsize=8, frameon=False)
save(fig, "ch28", "by_category.png")
print("part0 qty2500 test rows:", len(test[(test.part_id == 0) & (test.qty == 2500)]))
