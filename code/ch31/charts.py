"""charts.py - figures for chapter 30 (fictional data)."""
import sys; sys.path.insert(0, "../common")
import numpy as np, matplotlib.pyplot as plt
from fa_plot import setup, fa, save
from collections import Counter
from emails import EMAILS
from heldout import T
from extract import baseline_extract, scrub
from signals import update
setup()

def conf_of(pairs):
    m = np.zeros((4, 4), int)
    for g, p in pairs: m[g, p] += 1
    return m
dev = [(e["label"]["urgency"], baseline_extract(scrub(e["text"]))["urgency"]) for e in EMAILS]
ho = [(u, baseline_extract(scrub(t))["urgency"]) for t, u, *_ in T]
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, m, title in zip(axs, [conf_of(dev), conf_of(ho)],
                        ["ایمیل‌های توسعه (۸ تا)", "ایمیل‌های کنارگذاشته (۱۲ تا)"]):
    ax.imshow(m, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if m[i, j] > 2 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xlabel(fa("فوریت پیش‌بینی‌شده")); ax.set_ylabel(fa("فوریت برچسب انسانی"))
    ax.set_title(fa(title))
fig.tight_layout(); save(fig, "ch31", "urgency_confusion")

fields = ["urgency", "sensitivity", "tone", "competitor"]
names = ["فوریت", "حساسیت", "لحن", "رقیب"]
def acc(rows, gold_of, text_of):
    c = Counter()
    for r in rows:
        o = baseline_extract(scrub(text_of(r))); g = gold_of(r)
        for f in fields: c[f] += o[f] == g[f]
    return [c[f] / len(rows) for f in fields]
a1 = acc(EMAILS, lambda e: e["label"], lambda e: e["text"])
a2 = acc(T, lambda t: dict(zip(fields, t[1:])), lambda t: t[0])
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(4)
ax.bar(x - 0.2, a1, 0.4, label=fa("توسعه"), color="#9db8d6")
ax.bar(x + 0.2, a2, 0.4, label=fa("کنارگذاشته"), color="#d6806b")
ax.set_xticks(x); ax.set_xticklabels([fa(n) for n in names])
ax.set_ylim(0, 1.15); ax.set_ylabel(fa("دقت (Accuracy)"))
ax.legend(loc="upper right")
ax.set_title(fa("واژه‌نامه روی داده‌ای که دیده عالی است، روی داده‌ی تازه نه"))
save(fig, "ch31", "dev_vs_heldout")

fig, ax = plt.subplots(figsize=(8, 4))
p1 = update(0.55, ["price", "price", "both", "price"])
p2 = update(0.30, ["lead_time", "lead_time", "none"])
ax.plot(range(len(p1)), p1, "o-", label="NordSensor Example AG", color="#d6806b")
ax.plot(range(len(p2)), p2, "s-", label="AlpenBoard Example GmbH", color="#4a7bb7")
ax.set_xlabel(fa("تعداد ایمیل‌های خوانده‌شده")); ax.set_ylabel(fa("احتمال حساس‌بودن به قیمت"))
ax.set_ylim(0, 1); ax.set_xticks(range(5)); ax.grid(alpha=.3); ax.legend()
ax.set_title(fa("سابقه‌ی خرید نقطه‌ی شروع است؛ هر ایمیل کمی جابه‌جایش می‌کند"))
save(fig, "ch31", "posterior_update")

fig, ax = plt.subplots(figsize=(8, 3.6))
items = [("متن کامل ایمیل خام", 30), ("متن پاک‌شده از PII", 180),
         ("JSON سیگنال‌ها", 730), ("برچسب انسانی برای ارزیابی", 730),
         ("لاگ پرامپت/خروجی LLM", 14)]
for i, (n, d) in enumerate(items):
    ax.barh(i, d, color="#4a7bb7" if d > 100 else "#d6806b")
    ax.text(d + 10, i, fa(f"{d} روز", digits=True), va="center")
ax.set_yticks(range(len(items))); ax.set_yticklabels([fa(n) for n, _ in items])
ax.set_xlim(0, 850); ax.set_xlabel(fa("مدت نگهداری (روز) — نمونه، نه توصیه‌ی حقوقی"))
ax.invert_yaxis(); fig.tight_layout(); save(fig, "ch31", "retention")

ok = Counter(); tot = Counter()
for e in EMAILS:
    o = baseline_extract(scrub(e["text"]))
    tot[e["lang"]] += 1; ok[e["lang"]] += o["tone"] == e["label"]["tone"]
for t in T:
    l = "de" if any(w in t[0] for w in ("Bitte", "Danke", "Ihre", "Wir")) else "en"
    tot[l + "_h"] += 1; ok[l + "_h"] += baseline_extract(t[0])["tone"] == t[3]
keys = ["en", "de", "en_h", "de_h"]
labs = ["انگلیسی\nتوسعه", "آلمانی\nتوسعه", "انگلیسی\nکنارگذاشته", "آلمانی\nکنارگذاشته"]
vals = [ok[k] / tot[k] for k in keys]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(range(4), vals, color=["#9db8d6", "#9db8d6", "#d6806b", "#d6806b"])
for i, k in enumerate(keys):
    ax.text(i, vals[i] + .03, f"{ok[k]}/{tot[k]}", ha="center")
ax.set_xticks(range(4)); ax.set_xticklabels([fa(l) for l in labs])
ax.set_ylim(0, 1.15); ax.set_ylabel(fa("دقت لحن"))
ax.set_title(fa("دقت را جدا برای هر زبان بسنج"))
fig.tight_layout(); save(fig, "ch31", "tone_by_lang")
print({k: f"{ok[k]}/{tot[k]}" for k in keys})
