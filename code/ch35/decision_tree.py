"""decision_tree.py - draws the chapter 33 decision ladder (figure)."""
import sys; sys.path.insert(0, "../common")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from fa_plot import setup, fa, save

setup()
rows = [("جواب، یک واقعیت دقیق از داده‌ی امروز است؟", "ابزار و SQL، از لایه‌ی ابزار امن", "#9db8d6"),
        ("جواب، عددی پیش‌بینی‌شده از تاریخچه است؟", "مدل پیش‌بینی کوچک، با بازه", "#9db8d6"),
        ("جواب در یک سند است و منبع لازم است؟", "بازیابی با RAG، با ارجاع", "#9db8d6"),
        ("مشکل، رفتار است؟ قالب، لحن، سبک، دستور", "مسئله را دوباره تعریف کن", "#dddddd"),
        ("آیا Prompt خوب، چند مثال و خروجی ساختاریافته کافی است؟", "ارزان‌ترین راه: همان Prompt", "#7fb27f"),
        ("صدها نمونه‌ی تأییدشده و مجموعه‌ی ارزیابی داری؟", "اول داده و Eval بساز (فصل ۳۴ و ۳۶)", "#e0a458")]
fig, ax = plt.subplots(figsize=(8, 7.2)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(-1.6, len(rows) * 1.2 + .2)
def box(x, y, w, h, text, col, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=col, ec="#555555", lw=.8))
    ax.text(x + w / 2, y + h / 2, fa(text, True), ha="center", va="center", fontsize=size)
for i, (q, a, col) in enumerate(rows):
    y = (len(rows) - 1 - i) * 1.2 + .3
    box(4.4, y, 5.5, .75, q, "#f4f4f4")                   # question (right side, RTL)
    box(0.1, y, 3.4, .75, a, col, 9)                      # outcome (left side)
    yes = "نه" if i in (3, 5) else "بله"
    ax.annotate("", xy=(3.55, y + .37), xytext=(4.35, y + .37),
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(3.95, y + .5, fa(yes), ha="center", fontsize=8, color="#333333")
    nxt = "بله" if i in (3, 5) else "نه"
    ax.annotate("", xy=(7.15, y - (.55 if i == 5 else .4)), xytext=(7.15, y - .02),
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(7.4, y - .25, fa(nxt), ha="center", fontsize=8, color="#333333")
box(4.4, -1.2, 5.5, .8, "حالا وقت Fine-tuning با LoRA است (فصل ۳۵)", "#4a7bb7", 11)
ax.set_title(fa("درخت تصمیم: هر سؤال را از بالا به پایین بپرس"), fontsize=11)
save(fig, "ch35", "decision_tree.png")
