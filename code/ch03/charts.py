"""charts.py - Chapter 3 figures (illustrative, assumed numbers)."""
import matplotlib.pyplot as plt
from fa_plot import fa, save

# 1) Expected annual cost: SaaS without control vs private (assumed numbers)
labels = [fa("SaaS بدون کنترل", digits=False), fa("زیرساخت خصوصی")]
risk = [0.02 * 2_000_000, 0.005 * 2_000_000]
infra = [0, 60_000]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(labels, risk, color="#c0504d", label=fa("هزینه‌ی مورد انتظار ریسک"))
ax.bar(labels, infra, bottom=risk, color="#1f4e79",
       label=fa("هزینه‌ی اضافه‌ی زیرساخت"))
for i, (r, c) in enumerate(zip(risk, infra)):
    ax.text(i, r + c + 1500, fa(f"{(r + c) / 1000:.0f} هزار یورو"), ha="center")
ax.set_ylabel(fa("یورو در سال"))
ax.set_title(fa("مقایسه‌ی هزینه‌ی مورد انتظار (عددهای فرضی)"))
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, 85_000)
save(fig, "ch03", "risk_cost.png")

# 2) Sensitivity: break-even as the incident probability changes
import numpy as np
p = np.linspace(0, 0.06, 61)
saas = p * 2_000_000
private = 0.25 * p * 2_000_000 + 60_000   # assume private cuts risk by 75%
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(p * 100, saas / 1000, color="#c0504d", lw=2, label=fa("SaaS", digits=False))
ax.plot(p * 100, private / 1000, color="#1f4e79", lw=2, label=fa("خصوصی"))
be = 60_000 / (0.75 * 2_000_000)
ax.axvline(be * 100, color="gray", ls="--")
ax.text(be * 100 + 0.1, 100, fa(f"نقطه‌ی سربه‌سر ≈ {be*100:.0f}٪"))
ax.set_xlabel(fa("احتمال سالانه‌ی حادثه (٪)"))
ax.set_ylabel(fa("هزار یورو در سال"))
ax.set_title(fa("حساسیت تصمیم به برآورد احتمال (فرضی)"))
ax.legend(fontsize=8)
save(fig, "ch03", "risk_sensitivity.png")

# 3) Deployment options: qualitative control vs. operational effort (1-5 scale)
opts = {"ابر خصوصی اروپایی": (1.5, 1.5), "سرور اختصاصی": (2.5, 2.3),
        "Colocation": (3.6, 3.4), "On-prem": (4.3, 4.2),
        "Air-gapped": (5.0, 5.0)}
fig, ax = plt.subplots(figsize=(6.5, 4))
for name, (ctrl, eff) in opts.items():
    ax.scatter(eff, ctrl, s=160, color="#1f4e79")
    ax.annotate(fa(name, digits=False), (eff, ctrl), (eff - 0.2, ctrl + 0.25),
                ha="right")
ax.set_xlim(-0.3, 5.6); ax.set_ylim(0.5, 5.6)
ax.set_xlabel(fa("زحمت عملیاتی (کیفی، ۱ تا ۵)"))
ax.set_ylabel(fa("کنترل و جداسازی (کیفی، ۱ تا ۵)"))
ax.set_title(fa("گزینه‌های استقرار: کنترل در برابر زحمت (ارزیابی کیفی)"))
save(fig, "ch03", "deployment_tradeoff.png")
