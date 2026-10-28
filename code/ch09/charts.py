"""Chapter 9 charts (synthetic data). Run: python3 charts.py"""
import numpy as np
import pandas as pd

from fa_plot import fa, plt, save
from demand import MONTHS, make
from backtest import features, gbm, train_rows, wide

BLUE, RED, GREY, GREEN = "#1f4e79", "#c0504d", "#7f7f7f", "#4f8a3a"
d = make()
FA_M = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت",
        "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]

# 1) components of one smooth part
s = d[d.part == "EX-PART-004"].set_index("month").qty.astype(float)
trend = s.rolling(12, center=True).mean()
detr = s / trend
season = detr.groupby(detr.index.month).mean()
fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=False)
axes[0].plot(s.index, s, color=BLUE, lw=1.5, label=fa("تقاضای ماهانه"))
axes[0].plot(trend.index, trend, color=RED, lw=2, label=fa("روند (میانگین متحرک ۱۲ ماهه)"))
axes[0].legend(fontsize=8); axes[0].set_title(fa("EX-PART-004: تقاضا و روند (مصنوعی)", False))
axes[1].bar(range(1, 13), season.values, color=GREEN)
axes[1].axhline(1, color="k", lw=0.8)
axes[1].set_xticks(range(1, 13), [fa(m) for m in FA_M], fontsize=7, rotation=45)
axes[1].set_title(fa("الگوی فصلی: نسبت هر ماه به روند"))
axes[2].plot(detr.index, detr / detr.index.month.map(season).values, color=GREY, lw=1)
axes[2].axhline(1, color="k", lw=0.8)
axes[2].set_title(fa("نویز: آنچه بعد از حذف روند و فصل می‌ماند"))
save(fig, "ch09", "series_components.png")

# 2) three kinds of demand
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharex=True)
for ax, part, title in zip(axes, ["EX-PART-004", "EX-PART-039", "EX-PART-057"],
                           ["هموار", "متناوب", "نامنظم و حجیم"]):
    x = d[d.part == part]
    ax.bar(x.month, x.qty, width=20, color=BLUE)
    ax.set_title(fa(f"{title} ({part})", False), fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
axes[0].set_ylabel(fa("تعداد در ماه"))
save(fig, "ch09", "demand_kinds.png")

# 3) seasonal naive by hand
y24 = np.array([400, 420, 470, 430, 410, 400, 390, 280, 450, 450, 430, 340])
y25 = np.array([420, 440, 500, 450, 420, 430, 400, 300, 470, 480, 440, 350])
naive = np.r_[y24[-1], y25[:-1]]
fig, ax = plt.subplots(figsize=(8, 3.3))
m = np.arange(1, 13)
ax.plot(m, y25, "o-", color="k", label=fa("واقعی ۲۰۲۵"))
ax.plot(m, y24, "s--", color=GREEN, label=fa("Seasonal naive = همان ماه ۲۰۲۴", False))
ax.plot(m, naive, "^:", color=RED, label=fa("Naive = ماه قبل", False))
ax.set_xticks(m, [fa(x) for x in FA_M], fontsize=7, rotation=45)
ax.set_ylabel(fa("تعداد")); ax.legend(fontsize=8)
ax.set_title(fa("MAE: فصلی ۱۹٫۲ در برابر ماه قبل ۵۷٫۵ (اعداد نمونه)", False))
save(fig, "ch09", "snaive_hand.png")

# 4) Croston by hand
sp = np.array([0, 0, 300, 0, 0, 0, 250, 0, 0, 350, 0, 0])
cro = [np.nan] * 3 + [100.0] * 4 + [90.62] * 3 + [95.57] * 2
ma3 = [np.nan] * 3 + [100, 100, 100, 0, 83.3, 83.3, 83.3, 116.7, 116.7]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(m, sp, color=GREY, alpha=0.6, label=fa("تقاضای واقعی"))
ax.step(m, cro, where="mid", color=BLUE, lw=2, label=fa("پیش‌بینی Croston", False))
ax.step(m, ma3, where="mid", color=RED, lw=1.5, ls="--",
        label=fa("میانگین متحرک ۳ ماهه"))
ax.axhline(75, color=GREEN, lw=1, ls=":", label=fa("میانگین ساده = ۷۵"))
ax.set_xticks(m, [fa(i) for i in m]); ax.set_xlabel(fa("ماه"))
ax.set_ylabel(fa("تعداد")); ax.legend(fontsize=7, ncol=2)
ax.set_title(fa("تقاضای متناوب: Croston آرام است، میانگین متحرک می‌پرد", False))
save(fig, "ch09", "croston_hand.png")

# 5) backtest results (from backtest_results.csv written by backtest.py)
r = pd.read_csv("backtest_results.csv")
def wape(g):
    return 100 * np.abs(g.y - g.p).sum() / g.y.sum()
order = ["naive", "seasonal naive", "moving avg 3", "croston sba", "moving avg 12",
         "ma12 x season", "lightgbm (l2)", "lightgbm global"]
labels = ["ماه قبل", "همان ماه پارسال", "میانگین ۳", "Croston", "میانگین ۱۲",
          "میانگین ۱۲×فصل", "LightGBM (L2)", "LightGBM (L1)"]
sm = r[r.kind == "smooth"].groupby("method").apply(wape)[order]
tot = r[r.kind == "intermittent"].groupby(["method", "origin", "part_i"])[["y", "p"]]\
    .sum().reset_index().groupby("method").apply(wape)[order]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, vals, title in [(axes[0], sm, "قطعه‌های هموار، ماه به ماه"),
                        (axes[1], tot, "قطعه‌های متناوب، جمع ۳ ماه")]:
    c = [GREEN if v == vals.min() else BLUE for v in vals]
    ax.barh([fa(l, False) for l in labels], vals.values, color=c)
    for i, v in enumerate(vals.values):
        ax.text(v + 1, i, fa(f"{v:.1f}"), va="center", fontsize=7)
    ax.set_title(fa(title), fontsize=9); ax.set_xlabel(fa("WAPE (درصد)", False))
    ax.tick_params(axis="y", labelsize=7)
axes[1].axvline(100, color=RED, ls=":", lw=1)
save(fig, "ch09", "backtest_wape.png")

# 6) fan chart for one smooth part, origin = 2025-09 (index 44)
Y, kind = wide()
T = 44
X, yv = train_rows(Y, T)
qs = {a: gbm(alpha=a).fit(X, yv) for a in (0.1, 0.5, 0.9)}
i = 3                                       # EX-PART-004
band = {a: [] for a in qs}
for h in (1, 2, 3):
    f, sc = features(Y, T, h)
    for a, mdl in qs.items():
        band[a].append(max(0, mdl.predict(f.iloc[[i]])[0]) * sc[i])
fut = MONTHS[T + 1: T + 4]
fig, ax = plt.subplots(figsize=(8, 3.3))
ax.plot(MONTHS[24: T + 1], Y[i, 24: T + 1], "o-", color="k", ms=3,
        label=fa("تاریخچه"))
ax.fill_between(fut, band[0.1], band[0.9], color=BLUE, alpha=0.25,
                label=fa("بازه‌ی P10 تا P90", False))
ax.plot(fut, band[0.5], "o--", color=BLUE, label=fa("P50", False))
ax.plot(fut, Y[i, T + 1: T + 4], "x", color=RED, ms=9, mew=2,
        label=fa("واقعی (بعداً دیده شد)"))
ax.axvline(MONTHS[T], color=GREY, ls=":")
ax.set_ylabel(fa("تعداد در ماه")); ax.legend(fontsize=7, ncol=2)
ax.set_title(fa("پیش‌بینی احتمالاتی سه ماه آینده از مبدأ سپتامبر ۲۰۲۵ (مصنوعی)"))
save(fig, "ch09", "fan_chart.png")
print("fan values:", {a: np.round(v, 0).tolist() for a, v in band.items()},
      "actual:", Y[i, T + 1: T + 4].tolist())
