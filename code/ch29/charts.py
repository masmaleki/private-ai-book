"""charts.py - figures for chapter 28 (synthetic data)."""
import sys; sys.path.insert(0, "../common")
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import beta
from fa_plot import fa, setup, save

setup()
po = pd.read_csv("demo_po_history.csv")
s1 = po[(po.supplier_id == 1) & (po.promised_days == 28)].actual_days

fig, ax = plt.subplots(figsize=(6.0, 2.9))
ax.hist(s1, bins=range(20, 50), color="#7fa7d1", edgecolor="white")
ax.axvline(28, color="#1f3a5f", lw=2)
ax.text(27.6, ax.get_ylim()[1] * 0.92, fa("قول: ۲۸ روز"), ha="right",
        fontsize=8, color="#1f3a5f")
for q, c in [(0.5, "#e0a43a"), (0.9, "#c9563c")]:
    v = s1.quantile(q)
    ax.axvline(v, color=c, ls="--")
    ax.text(v + 0.3, ax.get_ylim()[1] * 0.8, f"P{int(q*100)}={v:.0f}",
            fontsize=8, color=c)
ax.set_xlabel(fa("Lead Time واقعی (روز)")); ax.set_ylabel(fa("تعداد سفارش"))
ax.set_title(fa("تأمین‌کننده‌ی ۱، قول ۲۸ روز: توزیع واقعی (ساختگی)"), fontsize=10)
save(fig, "ch29", "leadtime_hist.png")

g = po.groupby("supplier_id").late.agg(["size", "sum"])
on = g["size"] - g["sum"]
a, b = on + 8, g["sum"] + 2
mean = a / (a + b)
lo, hi = beta.ppf(0.05, a, b), beta.ppf(0.95, a, b)
fig, ax = plt.subplots(figsize=(5.6, 2.8))
x = np.arange(len(g))
ax.bar(x, mean, color="#3d6a99")
ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="k",
            capsize=3)
# a new supplier with only 4 orders, 4 on time
a2, b2 = 4 + 8, 0 + 2
ax.bar([len(g)], [a2 / (a2 + b2)], color="#e0a43a")
ax.errorbar([len(g)], [a2 / (a2 + b2)],
            yerr=[[a2 / (a2 + b2) - beta.ppf(.05, a2, b2)],
                  [beta.ppf(.95, a2, b2) - a2 / (a2 + b2)]],
            fmt="none", ecolor="k", capsize=3)
ax.set_xticks(list(x) + [len(g)], [f"S{i}" for i in x] + [fa("جدید (۴ سفارش)")],
              fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel(fa("نرخ تحویل به‌موقع"))
ax.set_title(fa("امتیاز اعتبار تأمین‌کننده با بازه‌ی ۹۰٪ (ساختگی)"), fontsize=10)
save(fig, "ch29", "supplier_score.png")

train, test = po.iloc[:2400], po.iloc[2400:]
X = ["supplier_id", "promised_days", "tightness", "qty"]
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         random_state=42, verbose=-1, n_jobs=2)
clf.fit(train[X], train.late, categorical_feature=["supplier_id"])
p = clf.predict_proba(test[X])[:, 1]
bins = pd.cut(p, [0, .2, .4, .6, .8, 1.0])
c = pd.DataFrame({"b": bins, "p": p, "y": test.late.values}) \
      .groupby("b", observed=True).mean()
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.plot([0, 1], [0, 1], ls="--", color="gray")
ax.plot(c.p, c.y, "o-", color="#c9563c")
ax.set_xlabel(fa("احتمال پیش‌بینی‌شده‌ی تأخیر"))
ax.set_ylabel(fa("نرخ واقعی تأخیر"))
ax.set_title(fa("نمودار کالیبراسیون"), fontsize=10)
save(fig, "ch29", "late_calibration.png")

r = np.random.default_rng(7)
oh = np.arange(1000, 2301, 50)
L = r.choice(s1.values, size=20000)
dem = r.poisson(45 * L)
probs = [(dem > o).mean() for o in oh]
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.plot(oh, probs, color="#1f3a5f", lw=2)
ax.axhline(0.05, ls="--", color="#c9563c")
ax.text(2280, 0.08, fa("سطح ریسک قابل قبول ۵٪"), ha="right", fontsize=8,
        color="#c9563c")
ax.set_xlabel(fa("موجودی در دسترس (عدد)"))
ax.set_ylabel(fa("احتمال کمبود"))
save(fig, "ch29", "stockout_curve.png")

D = pd.read_csv("demo_daily_demand.csv", index_col=0, parse_dates=True)
w = D["seasonal"].resample("W").sum()
fig, ax = plt.subplots(figsize=(6.2, 2.7))
ax.plot(w.index, w.values, color="#3d6a99", lw=1)
ax.set_ylabel(fa("تقاضای هفتگی"))
ax.set_title(fa("قطعه‌ی فصلی: الگوی سالانه تکرار می‌شود (ساختگی)"), fontsize=10)
save(fig, "ch29", "seasonal_demand.png")

s = D["intermittent"].iloc[-365:]
sums = np.array([s.iloc[i:i + 30].sum() for i in range(0, 336)])
fig, ax = plt.subplots(figsize=(5.6, 2.7))
ax.hist(sums, bins=30, color="#e0a43a", edgecolor="white")
ax.set_xlabel(fa("تقاضای ۳۰ روزه (عدد)")); ax.set_ylabel(fa("تعداد پنجره"))
ax.set_title(fa("قطعه‌ی متناوب: نیمی از پنجره‌ها صفر، چندتایی خیلی بزرگ"),
             fontsize=10)
save(fig, "ch29", "intermittent_hist.png")
