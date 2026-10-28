"""charts.py - Chapter 5 figures (synthetic, illustrative data)."""
import numpy as np
import matplotlib.pyplot as plt
from fa_plot import fa, save

rng = np.random.default_rng(42)

# 1) 2-D vectors and angles
vecs = {"A": (3.0, 0.5), "B": (2.9, 0.75), "C": (0.5, 2.0)}
fig, ax = plt.subplots(figsize=(5, 4))
for name, (x, y) in vecs.items():
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2,
                                color="#c0504d" if name == "C" else "#1f4e79"))
    ax.text(x + 0.05, y + 0.05, fa(f"قطعه‌ی {name}"))
ax.set_xlim(0, 3.6); ax.set_ylim(0, 2.4)
ax.set_xlabel(fa("جریان خروجی (آمپر)"))
ax.set_ylabel(fa("فرکانس سوئیچینگ (مگاهرتز)"))
ax.set_title(fa("زاویه‌ی کوچک یعنی شباهت زیاد"))
save(fig, "ch05", "vectors_angle.png")

# 2) normal distribution with P10/P50/P90
mu, sd = 7.85, 0.40
x = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
pdf = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
p10, p90 = mu - 1.2816 * sd, mu + 1.2816 * sd
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(x, pdf, color="#1f4e79", lw=2)
ax.fill_between(x, pdf, where=(x >= p10) & (x <= p90), color="#1f4e79",
                alpha=0.2)
for q, lab in [(p10, "P10"), (mu, "P50"), (p90, "P90")]:
    ax.axvline(q, color="#c0504d", ls="--")
    ax.text(q, pdf.max() * 1.03, fa(f"{lab}\n{q:.2f}", digits=False),
            ha="center", fontsize=8)
ax.set_xlabel(fa("قیمت واحد (یورو)"))
ax.set_ylabel(fa("چگالی احتمال"))
ax.set_title(fa("توزیع نرمال: ۸۰٪ احتمال بین P10 و P90"), pad=24)
save(fig, "ch05", "normal_quantiles.png")

# 3) raw vs log prices (synthetic catalogue of 2000 parts)
prices = np.exp(rng.normal(loc=0.0, scale=1.8, size=2000))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
a1.hist(prices, bins=60, color="#7f9fbf")
a1.set_title(fa("قیمت خام: چند قطعه‌ی گران همه را له می‌کنند"))
a1.set_xlabel(fa("یورو"))
a2.hist(np.log(prices), bins=40, color="#1f4e79")
a2.set_title(fa("لگاریتم قیمت: شکل متقارن و قابل مدل‌سازی"))
a2.set_xlabel(fa("log(قیمت)", digits=False))
save(fig, "ch05", "log_prices.png")

# 4) gradient descent path on loss(w) = (w-3)^2
w_grid = np.linspace(-0.5, 5, 200)
path = [0.0]
for _ in range(8):
    path.append(path[-1] - 0.1 * 2 * (path[-1] - 3))
path = np.array(path)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(w_grid, (w_grid - 3) ** 2, color="#7f9fbf", lw=2)
ax.plot(path, (path - 3) ** 2, "o-", color="#c0504d")
for i in (0, 1, 2, 3):
    ax.text(path[i] + 0.15, (path[i] - 3) ** 2 + 0.4, fa(f"گام {i}"))
ax.set_xlabel("w")
ax.set_ylabel(fa("خطا (Loss)", digits=False))
ax.set_title(fa("Gradient Descent: هر گام در جهت مخالف شیب", digits=False))
save(fig, "ch05", "gradient_descent.png")

# 5) learning-rate comparison: trajectory of w (target w = 3)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for lr, color in [(0.02, "#7f9fbf"), (0.1, "#1f4e79"), (0.9, "#c0504d")]:
    w, ws = 0.0, []
    for _ in range(25):
        ws.append(w)
        w -= lr * 2 * (w - 3)
    ax.plot(ws, "o-", ms=3, color=color, lw=1.5, label=fa(f"نرخ یادگیری {lr}"))
ax.axhline(3, color="gray", ls="--")
ax.set_xlabel(fa("گام"))
ax.set_ylabel(fa("مقدار w (هدف: ۳)", digits=False))
ax.set_title(fa("اثر نرخ یادگیری: خیلی کوچک کند، خیلی بزرگ نوسانی"))
ax.legend(fontsize=8)
save(fig, "ch05", "learning_rates.png")

# 6) softmax with temperature
logits = np.array([2.0, 1.0, 0.1])
labels = [fa("توضیح قیمت"), fa("پاک‌سازی BOM", digits=False), fa("سؤال دیتاشیت")]
fig, ax = plt.subplots(figsize=(7, 3.4))
width = 0.25
for k, T in enumerate([0.5, 1.0, 2.0]):
    z = logits / T
    p = np.exp(z - z.max()); p /= p.sum()
    ax.bar(np.arange(3) + (k - 1) * width, p, width,
           label=fa(f"Temperature = {T}"))
ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels)
ax.set_ylabel(fa("احتمال"))
ax.set_title(fa("اثر Temperature بر خروجی Softmax", digits=False))
ax.legend(fontsize=8)
save(fig, "ch05", "softmax_temperature.png")
