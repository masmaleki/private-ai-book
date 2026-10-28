"""charts.py - Chapter 7 figures (synthetic demo_* data)."""
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fa_plot import fa, save
from features import drop_outliers, load, normalise, offers_in_eur

warnings.filterwarnings("ignore")
BLUE, RED, GREY, GREEN = "#1f4e79", "#c0504d", "#7f7f7f", "#4f8a3c"
t = load()
o_raw = offers_in_eur(t["demo_offers"], t["fx_rates"])
SLOPE = -0.0529
o = normalise(o_raw, SLOPE)

# 1) as-of timeline for one quote
q = t["demo_quote_lines"]
row = q[(q.part_id == 14) & (q.issued_at > "2024-06-01")].iloc[0]
tq = row.issued_at
g = o[(o.part_id == 14) & (o.observed_at > tq - pd.Timedelta(days=50))
      & (o.observed_at < tq + pd.Timedelta(days=25))]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.axvspan(tq - pd.Timedelta(days=30), tq, color=BLUE, alpha=0.12)
ax.axvspan(tq - pd.Timedelta(days=15), tq + pd.Timedelta(days=15), ymin=0, ymax=0.12,
           color=RED, alpha=0.35)
past, fut = g[g.observed_at < tq], g[g.observed_at >= tq]
ax.scatter(past.observed_at, past.norm_price, color=BLUE, zorder=3)
ax.scatter(fut.observed_at, fut.norm_price, color=GREY, marker="x", zorder=3)
last = past.iloc[-1]
ax.scatter([last.observed_at], [last.norm_price], s=160, facecolors="none",
           edgecolors=GREEN, lw=2, zorder=4)
ax.axvline(tq, color="k", ls="--")
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.04)
y0 = ax.get_ylim()[1]
ax.text(tq, y0, fa(" لحظه‌ی Quote (t)", digits=False), va="top")
ax.text(tq - pd.Timedelta(days=29), y0, fa("پنجره‌ی مجاز: ۳۰ روز قبل از t"), va="top",
        fontsize=8, color=BLUE)
ax.text(tq - pd.Timedelta(days=14), ax.get_ylim()[0], fa("پنجره‌ی نشت‌دار (مرکزی)"),
        va="bottom", fontsize=8, color=RED)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set_ylabel(fa("قیمت نرمال‌شده (یورو)"))
ax.set_title(fa("دایره‌ی سبز: آخرین Offer قبل از t؛ ضربدرها: آینده، ممنوع", digits=False),
             pad=14)
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.03)
save(fig, "ch07", "asof_timeline.png")

# 2) point-in-time vs centred rolling median for an EOL part
eol_ids = t["demo_parts"].query("lifecycle == 'eol'").part_id
pid = o[o.part_id.isin(eol_ids)].part_id.value_counts().index[0]
g = drop_outliers(o[o.part_id == pid]).set_index("observed_at").sort_index()
past30 = g["norm_price"].rolling("30D", closed="left").median()
centred = g["norm_price"].rolling(31, center=True, min_periods=5).median()
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.scatter(g.index, g.norm_price, s=8, color=GREY, alpha=0.6, label=fa("Offerها"))
ax.plot(past30.index, past30, color=BLUE, lw=2, label=fa("میانه‌ی ۳۰ روز گذشته (درست)"))
ax.plot(centred.index, centred, color=RED, lw=1.5, ls="--",
        label=fa("میانه‌ی مرکزی (نشت از آینده)"))
ax.legend(fontsize=8, loc="upper left")
ax.set_ylabel(fa("قیمت نرمال‌شده (یورو)"))
ax.set_title(fa(f"قطعه‌ی {pid} (مصنوعی، اعلام EOL): خط‌چین زودتر از واقعیت می‌پرد"))
save(fig, "ch07", "rolling_median.png")

# 3) FX: historical vs today's rate
fx = t["fx_rates"]
err = (fx.usd_per_eur / fx.usd_per_eur.iloc[-1] - 1) * 100
fig, axs = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
axs[0].plot(fx.day, fx.usd_per_eur, color=BLUE)
axs[0].set_ylabel(fa("دلار به ازای ۱ یورو"))
axs[0].set_title(fa("نرخ ارز مصنوعی و خطای تبدیل با نرخ «امروز»"))
axs[1].fill_between(fx.day, err, color=RED, alpha=0.4)
axs[1].axhline(0, color="k", lw=0.8)
axs[1].set_ylabel(fa("خطای قیمت (درصد)"))
save(fig, "ch07", "fx_effect.png")

# 4) quantity normalisation for one part
g = drop_outliers(o[(o.part_id == 14)])
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
axs[0].scatter(g.qty, g.price_eur, s=8, color=GREY)
axs[0].set_xscale("log")
axs[0].set_title(fa("قبل: قیمت واحد خام")); axs[0].set_xlabel(fa("تعداد"))
axs[0].set_ylabel(fa("قیمت (یورو)"))
axs[1].scatter(g.qty, g.norm_price, s=8, color=BLUE)
axs[1].set_xscale("log")
axs[1].set_title(fa("بعد: معادل ۱۰۰۰ عدد")); axs[1].set_xlabel(fa("تعداد"))
save(fig, "ch07", "qty_normalisation.png")

# 5) outliers: ratio to previous median
oo = o.sort_values(["observed_at", "offer_id"]).copy()
prev = oo.groupby("part_id")["norm_price"].transform(
    lambda s: s.rolling(20, min_periods=5).median().shift(1))
r = np.log10(oo.norm_price / prev).dropna()
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(r, bins=120, color=BLUE)
ax.set_yscale("log")
for v in (np.log10(5), np.log10(0.2)):
    ax.axvline(v, color=RED, ls="--")
ax.set_xlabel(fa("نسبت قیمت به میانه‌ی ۲۰ پیشنهاد قبلی، در مقیاس log10"))
ax.set_ylabel(fa("تعداد (مقیاس لگاریتمی)"))
ax.set_title(fa("داده‌ی پرت: خوشه‌ی کوچک نزدیک ۲ یعنی ۱۰۰ برابر"))
save(fig, "ch07", "outliers.png")

# 6) honest vs leaky evaluation (numbers from experiments.py)
names = ["خط مبنا", "A ویژگی سالم", "B نشت آینده\n(بک‌تست)", "B در تولید",
         "Boosting\nتقسیم تصادفی", "Boosting\nتقسیم گروهی", "Boosting\nتقسیم زمانی"]
vals = [3.46, 3.04, 2.87, 3.21, 2.68, 3.07, 4.18]
cols = [GREY, BLUE, "#d9a3a2", RED, "#d9a3a2", GREY, BLUE]
fig, ax = plt.subplots(figsize=(9, 3.6))
bars = ax.bar([fa(n) for n in names], vals, color=cols)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.05, fa(f"{v}٪"), ha="center",
            fontsize=8)
ax.set_ylabel(fa("MAPE (درصد)", digits=False)); ax.set_ylim(0, 4.8)
ax.tick_params(axis="x", labelsize=8)
ax.set_title(fa("صورتی: عددهای خوش‌بینانه‌ای که در تولید تکرار نمی‌شوند"))
save(fig, "ch07", "leakage_results.png")
print("charts done")
