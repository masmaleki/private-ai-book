"""charts.py - Chapter 8 figures (example and synthetic data)."""
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from fa_plot import fa, save
from models import FEATS, data

warnings.filterwarnings("ignore")
BLUE, RED, GREY, GREEN = "#1f4e79", "#c0504d", "#7f7f7f", "#4f8a3c"

# 1) SSE for every split threshold (hand example)
thr = ["۳۰", "۷۵", "۳۰۰", "۷۵۰", "۳۰۰۰"]
sse = [0.70, 0.3925, 0.04, 0.3925, 0.70]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.bar(thr, sse, color=[GREY, GREY, BLUE, GREY, GREY])
ax.axhline(1.0, color=RED, ls="--")
ax.text(4.4, 1.0, fa("قبل از شکستن: ۱٫۰"), va="bottom", ha="right", color=RED,
        fontsize=8)
for b_, v in zip(bars, sse):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.02, fa(f"{v}"), ha="center",
            fontsize=8)
ax.set_xlabel(fa("آستانه: تعداد کمتر از"))
ax.set_ylabel(fa("مجموع مربع خطا بعد از شکستن"))
ax.set_ylim(0, 1.15)
ax.set_title(fa("بهترین شکست: تعداد کمتر از ۳۰۰"))
save(fig, "ch08", "split_sse.png")

# 2) boosting rounds (hand example)
qty = np.array([10, 50, 100, 500, 1000, 5000])
y = np.array([2.0, 1.9, 1.8, 1.2, 1.1, 1.0])
F = {0: np.full(6, 1.5), 1: np.r_[[1.7] * 3, [1.3] * 3], 2: np.r_[[1.8] * 3, [1.2] * 3]}
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.scatter(qty, y, color="k", zorder=4, label=fa("داده"))
cols = {0: GREY, 1: "#7fa7cf", 2: BLUE}
for r, f in F.items():
    ax.step(qty, f, where="mid", color=cols[r], lw=2,
            label=fa(f"دور {r}: MSE = {((y - f) ** 2).mean():.4f}"))
ax.set_xscale("log")
ax.set_xlabel(fa("تعداد")); ax.set_ylabel(fa("قیمت واحد (یورو)"))
ax.legend(fontsize=8)
ax.set_title(fa("Boosting: هر دور، نیمی از باقی‌مانده را اصلاح می‌کند", digits=False))
save(fig, "ch08", "boosting_rounds.png")

# 3) extrapolation
rng = np.random.default_rng(8)
day = np.arange(0, 730)
price = 1.00 + 0.0004 * day + rng.normal(0, 0.02, len(day))
tr = day < 365
X = day.reshape(-1, 1)
lin = LinearRegression().fit(X[tr], price[tr])
rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1).fit(X[tr], price[tr])
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.scatter(day, price, s=4, color=GREY, alpha=0.5, label=fa("قیمت (مصنوعی)"))
ax.plot(day, lin.predict(X), color=BLUE, lw=2, label=fa("رگرسیون خطی"))
ax.plot(day, rf.predict(X), color=RED, lw=2, label=fa("جنگل تصادفی"))
ax.axvline(365, color="k", ls="--")
ax.text(370, 0.98, fa("داده‌ی آموزش تا اینجا"), fontsize=8)
ax.set_xlabel(fa("روز")); ax.set_ylabel(fa("قیمت (یورو)")); ax.legend(fontsize=8)
ax.set_title(fa("درخت‌ها نمی‌توانند از بزرگ‌ترین مقداری که دیده‌اند بالاتر بروند"))
save(fig, "ch08", "extrapolation.png")

# 4) model comparison (numbers from models.py)
names = ["خط مبنا", "خطی", "جنگل", "LightGBM", "خطی", "جنگل", "LightGBM"]
vals = [3.46, 3.04, 3.63, 4.07, 3.04, 2.79, 2.88]
colors = [GREY, BLUE, RED, RED, BLUE, GREEN, GREEN]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
bars = ax.bar(range(7), vals, color=colors)
ax.set_xticks(range(7), [fa(n, digits=False) for n in names], fontsize=8)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.05, fa(f"{v}٪"), ha="center",
            fontsize=8)
ax.axvline(0.5, color="k", lw=0.5); ax.axvline(3.5, color="k", lw=0.5)
ax.text(2, 4.45, fa("هدف: لگاریتم قیمت"), ha="center", fontsize=9)
ax.text(5, 4.45, fa("هدف: لگاریتم قیمت ÷ بازار"), ha="center", fontsize=9)
ax.set_ylim(0, 4.8); ax.set_ylabel(fa("MAPE آزمون (درصد)", digits=False))
ax.set_title(fa("با هدف نسبی، درخت‌ها از مدل خطی جلو می‌زنند"), pad=18)
save(fig, "ch08", "model_compare.png")

# 5) pinball loss
e = np.linspace(-2, 2, 200)  # error = actual - prediction
fig, ax = plt.subplots(figsize=(6.5, 3.3))
for q, c in [(0.1, RED), (0.5, GREY), (0.9, BLUE)]:
    ax.plot(e, np.maximum(q * e, (q - 1) * e), color=c, lw=2, label=fa(f"q = {q}"))
ax.set_xlabel(fa("خطا = واقعیت − پیش‌بینی"))
ax.set_ylabel(fa("جریمه"))
ax.legend(); ax.set_title(fa("تابع خطای چندک (Pinball): جریمه‌ی نامتقارن", digits=False))
save(fig, "ch08", "pinball.png")

# 6) quantile band for the most quoted part in the test period
train, test = data()
preds = {}
for a in (0.1, 0.5, 0.9):
    m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                          learning_rate=0.03, num_leaves=15, min_child_samples=20,
                          verbose=-1, random_state=0, n_jobs=2).fit(train[FEATS],
                                                                    train["rel"])
    preds[a] = np.exp(test.market + m.predict(test[FEATS]))
pid = test.part_id.value_counts().index[0]
s = test.part_id == pid
d = pd.DataFrame({"t": test.issued_at[s], "y": np.exp(test.y[s]),
                  "p10": preds[0.1][s], "p50": preds[0.5][s], "p90": preds[0.9][s],
                  "q": test.qty[s]})
d = d[(d.q > 500) & (d.q < 5000)].sort_values("t")   # similar quantities only
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.fill_between(d.t, d.p10, d.p90, color=BLUE, alpha=0.2, step="mid",
                label=fa("بازه‌ی P10 تا P90", digits=False))
ax.plot(d.t, d.p50, color=BLUE, drawstyle="steps-mid", label=fa("P50", digits=False))
inside = (d.y >= d.p10) & (d.y <= d.p90)
ax.scatter(d.t[inside], d.y[inside], color="k", s=14, zorder=3, label=fa("واقعی، داخل بازه"))
ax.scatter(d.t[~inside], d.y[~inside], color=RED, s=22, zorder=3,
           label=fa("واقعی، بیرون بازه"))
ax.set_ylabel(fa("قیمت Quote (یورو)")); ax.legend(fontsize=7, ncol=2)
ax.set_title(fa(f"قطعه‌ی {pid}، دوره‌ی آزمون، تعداد ۵۰۰ تا ۵۰۰۰ (مصنوعی)"))
save(fig, "ch08", "quantile_band.png")

# 7) SHAP for one quote
m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                      min_child_samples=20, verbose=-1, random_state=0,
                      n_jobs=2).fit(train[FEATS], train["rel"])
ex = shap.TreeExplainer(m)
sv = ex.shap_values(test[FEATS].iloc[[0]])[0]
order = np.argsort(np.abs(sv))[::-1][:6][::-1]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh([FEATS[i] for i in order], sv[order],
        color=[RED if v < 0 else BLUE for v in sv[order]])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(fa("سهم در خروجی مدل (واحد لگاریتمی)"))
ax.set_title(fa(f"SHAP برای Quote {int(test.quote_line_id.iloc[0])}", digits=False))
save(fig, "ch08", "shap_quote.png")
print("charts done")
