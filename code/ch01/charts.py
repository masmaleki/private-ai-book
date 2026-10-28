"""Chapter 1 charts. Run: python3 charts.py (uses tiny_lm.py's corpus)."""
import contextlib
import io
import random
import re
from collections import Counter

import numpy as np

from fa_plot import fa, plt, save

with contextlib.redirect_stdout(io.StringIO()):       # silence tiny_lm's demo prints
    import tiny_lm as lm

BLUE, RED, GREY, GREEN = "#1f4e79", "#c0504d", "#7f7f7f", "#4f8a3a"

# 1) the tiny model's "weights": next-word probabilities
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for ax, word in zip(axes, ["customer", "of"]):
    p = lm.next_word_probs(word)
    ax.bar(list(p), list(p.values()), color=BLUE)
    for i, v in enumerate(p.values()):
        ax.text(i, v + 0.02, fa(f"{v:.0%}".replace("%", "٪")), ha="center", fontsize=9)
    ax.set_ylim(0, 0.65)
    ax.set_title(fa(f"بعد از «{word}»", False), fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel(fa("احتمال کلمه‌ی بعدی"))
save(fig, "ch01", "tiny_lm_probs.png")

# 2) hallucinated "facts": which (quantity, MPN) pairs does the model generate?
text = " ".join(lm.corpus)
pat = re.compile(r"(\d+) pcs of (\w+)")
true_facts = set(pat.findall(text))
gen = Counter()
for seed in range(2000):
    gen.update(pat.findall(lm.generate("the", n_words=12, seed=seed)))
total = sum(gen.values())
false = sum(n for f, n in gen.items() if f not in true_facts)
print("training facts:", "; ".join(f"{n} pcs of {m}" for n, m in sorted(true_facts)))
print(f"generated 'N pcs of MPN' claims: {total}, not in training data: {false} "
      f"({false / total:.0%})")
items = sorted(gen.items(), key=lambda kv: -kv[1])
fig, ax = plt.subplots(figsize=(8, 3.4))
labels = [f"{q} pcs of {m}" for (q, m), _ in items]
colors = [GREEN if f in true_facts else RED for f, _ in items]
ax.barh(labels[::-1], [n for _, n in items][::-1], color=colors[::-1])
ax.set_xlabel(fa("تعداد دفعات در ۲۰۰۰ جمله‌ی تولیدشده"))
ax.tick_params(axis="y", labelsize=8)
ax.set_title(fa(f"سبز: در داده‌ی آموزش بود؛ قرمز: ساختگی ({false / total:.0%})"
                .replace("%", "٪")))
save(fig, "ch01", "tiny_lm_facts.png")

# 3) memory for model weights: params x bytes per weight (weights only)
sizes = [7, 14, 32, 70]
bits = {"۱۶ بیت": 16, "۸ بیت": 8, "۴ بیت": 4}
fig, ax = plt.subplots(figsize=(7.5, 3.4))
w = 0.26
for k, (name, b) in enumerate(bits.items()):
    gb = [s * b / 8 for s in sizes]
    xs = np.arange(len(sizes)) + (k - 1) * w
    ax.bar(xs, gb, w, label=fa(name), color=[BLUE, GREEN, RED][k])
    for x, g in zip(xs, gb):
        ax.text(x, g + 2, fa(f"{g:g}"), ha="center", fontsize=7)
ax.set_xticks(range(len(sizes)), [f"{s}B" for s in sizes])
ax.set_xlabel(fa("تعداد پارامتر (میلیارد)"))
ax.set_ylabel(fa("گیگابایت، فقط وزن‌ها"))
ax.legend(fontsize=8)
ax.set_title(fa("حجم وزن‌ها = تعداد پارامتر × بایت هر وزن"))
save(fig, "ch01", "model_memory.png")

# 4) Sarah's seven tasks vs the four building blocks (2 = main, 1 = helper)
tasks = ["فهمیدن ایمیل", "خواندن BOM", "اعتبارسنجی MPN", "موجودی و Offerها",
         "قطعه‌ی جایگزین", "قیمت‌گذاری", "نوشتن پاسخ"]
blocks = ["LLM خصوصی", "جست‌وجو و RAG", "مدل پیش‌بینی", "لایه‌ی ابزار امن"]
M = np.array([[2, 1, 0, 0],
              [2, 1, 0, 1],
              [1, 2, 0, 2],
              [0, 0, 0, 2],
              [1, 2, 0, 1],
              [1, 0, 2, 1],
              [2, 1, 0, 0]])
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.imshow(M, cmap="Blues", vmin=0, vmax=2.6, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i, j] == 2:
            ax.scatter(j, i, s=140, color="white", edgecolors=BLUE, linewidths=1.5)
            ax.scatter(j, i, s=70, color=BLUE)
        elif M[i, j] == 1:
            ax.scatter(j, i, s=110, facecolors="none", edgecolors=BLUE, linewidths=1.5)
ax.set_xticks(range(4), [fa(b, False) for b in blocks], fontsize=8)
ax.set_yticks(range(len(tasks)), [fa(t, False) for t in tasks], fontsize=8)
ax.grid(False)
ax.set_title(fa("کدام بلوک، کدام کار؟ دایره‌ی پر: نقش اصلی، توخالی: کمکی"))
save(fig, "ch01", "tasks_blocks.png")
