"""charts.py - figures for chapter 33 from the REAL toy runs
(toy_sft_result.json, facts_result.json, probe_result.json)."""
import sys; sys.path.insert(0, "../common")
import json
import numpy as np
import matplotlib.pyplot as plt
from fa_plot import setup, fa, save

setup()
t = json.load(open("toy_sft_result.json"))
f = json.load(open("facts_result.json"))
p = json.load(open("probe_result.json"))

def smooth(x, k=25):
    x = np.array(x); return np.convolve(x, np.ones(k) / k, mode="valid")

# 1) loss curves: pretraining then SFT
fig, ax = plt.subplots(figsize=(8, 3.8))
pre, sft = smooth(t["pre_losses"]), smooth(t["sft_losses"])
ax.plot(range(len(pre)), pre, color="#4a7bb7", label=fa("پیش‌آموزش روی متن عمومی"))
off = len(t["pre_losses"])
ax.plot(range(off, off + len(sft)), sft, color="#e0a458",
        label=fa("Fine-tuning روی ایمیل به JSON"))
ax.axvline(off, color="gray", ls="--", lw=.8)
ax.set_yscale("log"); ax.set_xlabel(fa("گام آموزش"))
ax.set_ylabel(fa("Loss (مقیاس لگاریتمی)")); ax.grid(alpha=.3); ax.legend()
ax.set_title(fa("مدل اسباب‌بازی: Loss هر مرحله روی داده‌ی همان مرحله"))
save(fig, "ch35", "toy_loss.png")

# 2) behaviour before / after
labels = ["مدل پایه", "پس از Fine-tuning", "Fine-tuning + تکرار متن عمومی"]
valid = [t["before"][0], t["after"][0], f["replay"]["valid"]]
exact = [t["before"][1], t["after"][1], f["replay"]["exact"]]
x = np.arange(3); w = .36
fig, ax = plt.subplots(figsize=(8, 3.8))
b1 = ax.bar(x - w / 2, valid, w, color="#9db8d6", label=fa("JSON معتبر"))
b2 = ax.bar(x + w / 2, exact, w, color="#4a7bb7", label=fa("کاملاً درست"))
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .02,
                fa(f"{b.get_height():.0%}", True), ha="center", fontsize=8)
ax.set_xticks(x, [fa(s) for s in labels]); ax.set_ylim(0, 1.15)
ax.set_ylabel(fa("سهم از ۶۰ ایمیل دیده‌نشده")); ax.legend(loc="upper left")
ax.set_title(fa("رفتار عوض شد: از ادامه دادن متن به JSON"))
save(fig, "ch35", "behaviour_before_after.png")

# 3) forgetting: loss on generic text
names = ["مدل پایه", "Fine-tuning\nایمیل به JSON", "Fine-tuning\n+ تکرار متن عمومی",
         "Fine-tuning\nروی قیمت‌ها"]
vals = [t["generic"][0], t["generic"][1], f["replay"]["generic_loss"],
        f["facts"]["generic_loss"]]
fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.bar([fa(n) for n in names], vals,
              color=["#7fb27f", "#d9534f", "#7fb27f", "#d9534f"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .05,
            fa(f"{b.get_height():.2f}", True), ha="center", fontsize=9)
ax.set_ylabel(fa("Loss روی متن عمومی (کمتر بهتر)"))
ax.set_title(fa("فراموشی فاجعه‌بار و درمانش با تکرار داده‌ی قدیمی"))
save(fig, "ch35", "forgetting.png")

# 4) facts experiment
known = f["facts"]["acc_known"]
un_n, un_d = map(int, f["facts"]["unseen_numeric"].split("/"))
st_n, st_d = map(int, f["facts"]["stale_after_change"].split("/"))
leak = [f["facts"]["leak_same_tenant"] == "0.0081",
        f["facts"]["leak_other_tenant_prompt"] == "0.0081"]
names = ["۵۰ قطعه‌ی آموزش:\nقیمت حفظ‌شده", "۱۰ قطعه‌ی ندیده:\nعدد ساختگی با اطمینان",
         "۱۰ قطعه پس از\nافزایش ۱۲٪: قیمت کهنه", "پرامپت Tenant دیگر:\nقیمت محرمانه لو رفت"]
vals = [known, un_n / un_d, st_n / st_d, float(leak[1])]
fig, ax = plt.subplots(figsize=(8.5, 3.9))
bars = ax.bar([fa(n) for n in names], vals,
              color=["#9db8d6", "#d9534f", "#d9534f", "#d9534f"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .02,
            fa(f"{b.get_height():.0%}", True), ha="center", fontsize=9)
ax.set_ylim(0, 1.15); ax.set_ylabel(fa("سهم"))
ax.set_title(fa("آموزش «واقعیت» با Fine-tuning: حفظ می‌کند، می‌سازد، کهنه می‌شود، لو می‌دهد"))
save(fig, "ch35", "facts_failure.png")

# 5) narrow behaviour: familiar vs new phrasing
names = ["جمله‌بندی‌های آشنا\n(۶۰ ایمیل)"] + [
    "kindly offer …", "MPN: … pcs needed", "price for … please"]
v = [t["after"][0]] + [a / 20 for a, _ in p["new_phrasing"].values()]
e = [t["after"][1]] + [b / 20 for _, b in p["new_phrasing"].values()]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(x - w / 2, v, w, color="#9db8d6", label=fa("JSON معتبر"))
ax.bar(x + w / 2, e, w, color="#4a7bb7", label=fa("کاملاً درست"))
for i in range(4):
    ax.text(i - w / 2, v[i] + .02, fa(f"{v[i]:.0%}", True), ha="center", fontsize=8)
    ax.text(i + w / 2, e[i] + .02, fa(f"{e[i]:.0%}", True), ha="center", fontsize=8)
ax.set_xticks(x, [fa(n) for n in names]); ax.set_ylim(0, 1.15); ax.legend()
ax.set_title(fa("رفتار یادگرفته‌شده به اندازه‌ی تنوع داده‌ی آموزشی است"))
save(fig, "ch35", "narrow_behaviour.png")

# 6) how many examples? memorise vs copy (data_curve.py, data_curve_diag.py)
dc = json.load(open("data_curve_result.json"))
dd = json.load(open("data_curve_diag.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
ns = list(dc.keys())
a1.plot(range(len(ns)), [dc[n]["valid"] for n in ns], "o-", color="#9db8d6",
        label=fa("JSON معتبر"))
a1.plot(range(len(ns)), [dc[n]["exact"] for n in ns], "s-", color="#4a7bb7",
        label=fa("کاملاً درست"))
a1.set_xticks(range(len(ns)), [fa(n, True) for n in ns]); a1.set_ylim(0, 1.1)
a1.set_xlabel(fa("تعداد نمونه‌ی SFT")); a1.legend(); a1.grid(alpha=.3)
a1.set_title(fa("قالب زود یاد گرفته می‌شود، کار دیر"), fontsize=9)
keys = [("from_email", "کپی از ایمیل (درست)", "#4a7bb7"),
        ("from_training_set", "MPN حفظ‌شده از آموزش", "#d9534f"),
        ("other", "رشته‌ی ساختگی", "#e0a458"), ("invalid", "JSON نامعتبر", "#bbbbbb")]
bottom = np.zeros(len(dd))
for k, lab, col in keys:
    v = np.array([dd[n][k] for n in dd])
    a2.bar([fa(n, True) for n in dd], v, bottom=bottom, color=col, label=fa(lab))
    bottom += v
a2.set_xlabel(fa("تعداد نمونه‌ی SFT")); a2.set_ylabel(fa("از ۶۰ ایمیل آزمون"))
a2.set_ylim(0, 84); a2.legend(fontsize=7, loc="upper center", ncol=2)
a2.set_title(fa("MPN خروجی از کجا آمد؟"), fontsize=9)
save(fig, "ch35", "data_curve.png")
