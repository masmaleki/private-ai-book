"""charts.py - Chapter 6 figures (synthetic, illustrative data)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from fa_plot import fa, save

BLUE, RED, GREY, GREEN = "#1f4e79", "#c0504d", "#7f7f7f", "#4f8a3c"

# 1) five points and the fitted line
qty = np.array([1, 10, 100, 1000, 10000]); y = np.array([1.20, 1.02, 0.86, 0.70, 0.51])
x = np.log10(qty); a, b = 1.198, -0.170
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
xx = np.linspace(0, 4.2, 100)
axs[0].scatter(qty, y, color=BLUE, zorder=3)
axs[0].plot(10 ** xx, a + b * xx, color=RED)
axs[0].set_xlabel(fa("تعداد (مقیاس خطی)")); axs[0].set_ylabel(fa("قیمت واحد (یورو)"))
axs[0].set_title(fa("روی محور خطی: منحنی"))
axs[1].scatter(x, y, color=BLUE, zorder=3)
axs[1].plot(xx, a + b * xx, color=RED)
axs[1].scatter([np.log10(500)], [a + b * np.log10(500)], marker="x", s=80, color=GREEN,
               zorder=4)
axs[1].annotate(fa("پیش‌بینی برای ۵۰۰ عدد: ۰٫۷۳۹", digits=False),
                (np.log10(500), 0.739), (0.4, 0.6), arrowprops=dict(arrowstyle="->"))
axs[1].set_xlabel(fa("log10 تعداد", digits=False)); axs[1].set_title(fa("روی محور لگاریتمی: خط"))
save(fig, "ch06", "five_points_fit.png")

# 2) gradient descent convergence on the five points
a_, b_, lr, hist = 0.0, 0.0, 0.05, []
for _ in range(600):
    err = a_ + b_ * x - y
    a_, b_ = a_ - lr * 2 * err.mean(), b_ - lr * 2 * (err * x).mean()
    hist.append((a_, b_, ((a_ + b_ * x - y) ** 2).mean()))
h = np.array(hist)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].semilogy(h[:, 2], color=BLUE)
axs[0].set_xlabel(fa("گام")); axs[0].set_ylabel(fa("MSE (مقیاس لگاریتمی)", digits=False))
axs[0].set_title(fa("خطا با هر گام کم می‌شود"))
axs[1].plot(h[:, 0], color=BLUE, label=fa("a (عرض از مبدأ)"))
axs[1].plot(h[:, 1], color=RED, label=fa("b (شیب)"))
axs[1].axhline(1.198, color=BLUE, ls=":"); axs[1].axhline(-0.170, color=RED, ls=":")
axs[1].set_xlabel(fa("گام")); axs[1].legend(); axs[1].set_title(fa("وزن‌ها به جواب بسته می‌رسند"))
save(fig, "ch06", "gd_convergence.png")

# 3) MSE vs MAE as a function of the error
e = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(e, e ** 2, color=BLUE, label=fa("مربع خطا (MSE)", digits=False))
ax.plot(e, np.abs(e), color=RED, label=fa("قدرمطلق خطا (MAE)", digits=False))
ax.set_xlabel(fa("خطا = پیش‌بینی − واقعیت")); ax.set_ylabel(fa("جریمه"))
ax.legend(); ax.set_title(fa("MSE خطاهای بزرگ را خیلی بیشتر جریمه می‌کند", digits=False))
save(fig, "ch06", "mse_vs_mae.png")

# 4) temporal split on the synthetic data, one part
q = pd.read_csv("data/quotes.csv", parse_dates=["quote_date"])
top = q.part_id.value_counts().index[0]
one = q[q.part_id == top]
cut = pd.Timestamp("2025-07-01")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.scatter(one.quote_date[one.quote_date < cut], one.unit_price_eur[one.quote_date < cut],
           s=12, color=BLUE, label=fa("آموزش (Train)", digits=False))
ax.scatter(one.quote_date[one.quote_date >= cut], one.unit_price_eur[one.quote_date >= cut],
           s=12, color=RED, label=fa("آزمون (Test)", digits=False))
ax.axvline(cut, color="k", ls="--")
ax.text(cut, ax.get_ylim()[1], fa(" نقطه‌ی برش"), va="top")
ax.set_ylabel(fa("قیمت واحد (یورو)")); ax.legend(loc="lower left")
ax.set_title(fa(f"تقسیم زمانی برای قطعه‌ی {top} (داده‌ی مصنوعی)", digits=False))
save(fig, "ch06", "temporal_split.png")

# 5) baselines vs model (numbers from train_linear.py run)
names = ["آخرین قیمت", "میانه‌ی قطعه", "مدل خطی", "مدل خطی\n(تقسیم تصادفی)"]
mapes = [13.6, 15.9, 9.6, 6.9]
fig, ax = plt.subplots(figsize=(7, 3.4))
bars = ax.bar([fa(n) for n in names], mapes, color=[GREY, GREY, BLUE, "#d9a3a2"])
for r, v in zip(bars, mapes):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.3, fa(f"{v}٪"), ha="center")
ax.set_ylabel(fa("MAPE (درصد)", digits=False)); ax.set_ylim(0, 19)
ax.set_title(fa("خطای آزمون: ستون صورتی خوش‌بینانه و گمراه‌کننده است"))
save(fig, "ch06", "baseline_vs_model.png")

# 6) under- and overfitting
rng = np.random.default_rng(1)
xs = np.sort(rng.uniform(0, 4, 10)); ys = 1.2 - 0.17 * xs + rng.normal(0, 0.04, 10)
grid = np.linspace(0, 4, 300)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
titles = {0: "درجه‌ی ۰: کم‌برازش", 1: "درجه‌ی ۱: مناسب", 9: "درجه‌ی ۹: بیش‌برازش"}
for ax, d in zip(axs, (0, 1, 9)):
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xs[:, None], ys)
    ax.scatter(xs, ys, color=BLUE, zorder=3)
    ax.plot(grid, m.predict(grid[:, None]), color=RED)
    ax.set_ylim(0.4, 1.5); ax.set_title(fa(titles[d], digits=False))
    ax.set_xlabel(fa("log10 تعداد", digits=False))
axs[0].set_ylabel(fa("قیمت (یورو)"))
save(fig, "ch06", "overfit_poly.png")
print("charts done")
