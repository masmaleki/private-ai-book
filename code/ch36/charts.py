"""charts.py - figures for chapter 34 from the REAL pipeline outputs."""
import sys; sys.path.insert(0, "../common")
import json
import numpy as np
import matplotlib.pyplot as plt
from fa_plot import setup, fa, save

setup()
st = json.load(open("pipeline_stats.json"))
au = json.load(open("augment_stats.json"))
co = json.load(open("contamination.json"))
FA_NAMES = {"raw export": "خروجی خام", "defence tenant (must never be here)": "Tenant-D (نباید اینجا باشد)",
            "no consent / opted out": "بدون مجوز / انصراف", "not approved (draft)": "تأییدنشده (پیش‌نویس)",
            "empty or too short reply": "پاسخ خالی یا خیلی کوتاه",
            "reply language != email language": "زبان پاسخ ≠ زبان ایمیل",
            "internal info in reply": "اطلاعات داخلی در پاسخ",
            "label does not match email": "برچسب با ایمیل نمی‌خواند",
            "exact duplicate": "تکراری دقیق", "near duplicate (Jaccard >= 0.8)": "تقریباً تکراری",
            "duplicate after pseudonymisation": "تکراری پس از جایگزینی PII"}

# 1) funnel
names = [FA_NAMES[n] for n, _ in st["funnel"]][::-1]
vals = [v for _, v in st["funnel"]][::-1]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(range(len(vals)), vals, color=["#4a7bb7"] + ["#9db8d6"] * (len(vals) - 2) + ["#bbbbbb"])
for i, v in enumerate(vals):
    ax.text(v + 5, i, fa(str(v), True), va="center", fontsize=8)
ax.set_yticks(range(len(vals)), [fa(n) for n in names], fontsize=8)
ax.set_xlabel(fa("تعداد جفت ایمیل و پاسخ باقی‌مانده"))
ax.set_title(fa(f"قیف پاک‌سازی: از {vals[-1]} جفت خام تا {vals[0]} جفت قابل استفاده", True))
save(fig, "ch36", "funnel.png")

# 2) near-duplicate similarity
s = np.array(st["near_dup_similarity"])
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(s, bins=np.arange(0.2, 1.01, 0.025), color="#9db8d6", edgecolor="white")
ax.axvline(0.8, color="#d9534f", ls="--"); ax.text(0.805, ax.get_ylim()[1] * .85,
                                                  fa("آستانه‌ی ۰٫۸"), color="#d9534f")
ax.set_xlabel(fa("بیشترین شباهت Jaccard با نمونه‌های نگه‌داشته‌شده‌ی قبلی"))
ax.set_ylabel(fa("تعداد"))
ax.set_title(fa("بیشتر نمونه‌ها شبیه‌اند اما تکراری نیستند؛ دمِ راست، کپی‌های Forward است"))
save(fig, "ch36", "near_dup_hist.png")

# 3) PII and facts replaced
pii = st["pii_replaced"]
lab = {"phone": "تلفن", "name": "نام شخص", "email": "ایمیل", "company": "نام شرکت",
       "staff": "نام کارمند ما", "iban": "IBAN", "fact": "قیمت و Lead Time"}
keys = ["phone", "name", "email", "company", "staff", "fact"]
fig, ax = plt.subplots(figsize=(8, 3.6))
bars = ax.bar([fa(lab[k]).replace("←", "به") for k in keys], [pii[k] for k in keys],
              color=["#e0a458"] * 5 + ["#4a7bb7"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 10,
            fa(str(int(b.get_height())), True), ha="center", fontsize=8)
ax.set_ylabel(fa("تعداد جایگزینی"))
ax.set_title(fa("جایگزینی داده‌ی شخصی و واقعیت‌ها؛ PII باقی‌مانده پس از بررسی: ۰"))
save(fig, "ch36", "pii_counts.png")

# 4) coverage before / after synthetic augmentation (train, extraction task)
cells = ["en/lead_time", "en/price", "en/unknown", "de/lead_time", "de/price", "de/unknown"]
x = np.arange(len(cells)); w = .38
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(x - w / 2, [au["before"][c] for c in cells], w, color="#9db8d6", label=fa("فقط داده‌ی واقعی"))
ax.bar(x + w / 2, [au["after"][c] for c in cells], w, color="#4a7bb7",
       label=fa("واقعی + مصنوعی (فقط train)"))
ax.axhline(60, color="gray", ls=":", lw=1)
ax.set_xticks(x, cells, fontsize=8); ax.legend(fontsize=8)
ax.set_ylabel(fa("نمونه در train"))
ax.set_title(fa("پوشش زبان × اولویت در داده‌ی آموزش استخراج"))
save(fig, "ch36", "coverage.png")

# 5) contamination
fig, ax = plt.subplots(figsize=(7, 3.4))
lbl = ["بدون حذف تکراری،\nتقسیم تصادفی ردیف‌ها", "با حذف تکراری،\nتقسیم بر اساس Thread"]
v = [co["no_dedup_random"], co["dedup_thread"]]
bars = ax.bar([fa(l) for l in lbl], v, color=["#d9534f", "#7fb27f"], width=.5)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .005,
            fa(f"{b.get_height():.0%}", True), ha="center")
ax.set_ylabel(fa("دقت «مدل حفظ‌کننده» روی test")); ax.set_ylim(0, .26)
ax.set_title(fa("مدلی که هیچ چیز یاد نگرفته، روی test آلوده نمره می‌گیرد"))
save(fig, "ch36", "contamination.png")
