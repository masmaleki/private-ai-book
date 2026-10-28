"""charts.py - Chapter 2 figures (fictional example data)."""
import numpy as np
import matplotlib.pyplot as plt
from fa_plot import fa, save

# 1) Price breaks: unit price and total cost vs. quantity (TPS5430DDAR example)
breaks = [(1, 2.10), (100, 1.62), (1000, 1.21), (2500, 1.08)]
def unit(q):
    p = None
    for t, v in breaks:
        if q >= t:
            p = v
    return p
q = np.arange(1, 3001)
u = np.array([unit(x) for x in q])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.step(q, u, where="post", color="#1f4e79", lw=2)
a1.set_xscale("log")
a1.set_xlabel(fa("تعداد (مقیاس لگاریتمی)"))
a1.set_ylabel(fa("قیمت واحد (یورو)"))
a1.set_title(fa("قیمت واحد پله‌ای"))
a2.plot(q, q * u, color="#c0504d", lw=2)
a2.axvline(800, color="gray", ls="--")
a2.scatter([800, 1000], [800 * 1.62, 1000 * 1.21], color="black", zorder=3)
a2.annotate(fa("۸۰۰ عدد: ۱۲۹۶ یورو"), (800, 1296), (150, 2300),
            arrowprops=dict(arrowstyle="->"))
a2.annotate(fa("۱۰۰۰ عدد: ۱۲۱۰ یورو"), (1000, 1210), (1300, 600),
            arrowprops=dict(arrowstyle="->"))
a2.set_xlabel(fa("تعداد خرید"))
a2.set_ylabel(fa("هزینه‌ی کل (یورو)"))
a2.set_title(fa("هزینه‌ی کل: پرش از پله"))
save(fig, "ch02", "price_breaks.png")

# 2) Effective unit cost for a need of 800 pcs under three buying options
labels = [fa("دقیقاً ۸۰۰"), fa("پله‌ی ۱۰۰۰"), fa("قرقره‌ی ۲۵۰۰")]
eff = [1296 / 800, 1210 / 800, 2700 / 800]
fig, ax = plt.subplots(figsize=(6, 3.2))
bars = ax.bar(labels, eff, color=["#7f9fbf", "#1f4e79", "#c0504d"])
for b, v in zip(bars, eff):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, fa(f"{v:.2f}"),
            ha="center")
ax.set_ylabel(fa("قیمت مؤثر هر قطعه‌ی لازم (یورو)"))
ax.set_title(fa("قیمت مؤثر برای نیاز ۸۰۰ عدد"))
save(fig, "ch02", "effective_cost.png")

# 3) Illustrative lead-time regime shift (synthetic, NOT real market data)
rng = np.random.default_rng(42)
months = np.arange(48)
base = 10 + 0.5 * rng.standard_normal(48)
shock = np.where((months >= 14) & (months < 34),
                 22 * np.sin(np.pi * (months - 14) / 20), 0)
lt = base + shock
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(months, lt, color="#1f4e79", lw=2)
ax.axvspan(14, 34, color="#c0504d", alpha=0.12)
ax.text(24, 14, fa("رژیم کمبود (ساختگی)"), ha="center")
ax.set_xlabel(fa("ماه"))
ax.set_ylabel(fa("Lead Time میانه (هفته)", digits=False))
ax.set_title(fa("تغییر رژیم بازار: داده‌ی مصنوعی برای نمایش مفهوم"))
save(fig, "ch02", "regime_shift.png")

# 4) Share of BOM lines by cleaning outcome (the 12-line example)
outcome = {"قطعه‌ی یکتا": 8, "ادغام‌شده (تکراری)": 2, "DNP": 1,
           "ناقص: پرسش از مشتری": 1}
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh([fa(k) for k in outcome], list(outcome.values()), color="#1f4e79")
for i, v in enumerate(outcome.values()):
    ax.text(v + 0.1, i, fa(v), va="center")
ax.set_xlabel(fa("تعداد ردیف"))
ax.set_title(fa("سرنوشت ۱۲ ردیف BOM نمونه"))
save(fig, "ch02", "bom_outcome.png")
