"""diagrams.py - draws the two appendix D figures (lifecycle, release gate)."""
import sys; sys.path.insert(0, "../common")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from fa_plot import setup, fa, save

setup()


def box(ax, x, y, w, h, text, col, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=col, ec="#555555", lw=.8))
    ax.text(x + w / 2, y + h / 2, fa(text, True), ha="center", va="center",
            fontsize=size, linespacing=1.6)


def arrow(ax, a, b, style="-|>", ls="-"):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle=style, lw=1.1, ls=ls, color="#333333"))


def lifecycle():
    # stages: (title, template); snake layout, top row right-to-left
    st = [("ایده", "درخواست ادمیر"), ("تعریف", "برگه‌ی یک‌صفحه‌ای"),
          ("داده", "کارت دیتاست و DPIA"), ("ساخت", "مدل تهدید"),
          ("ارزیابی", "کارت مدل"), ("انتشار", "فهرست دروازه"),
          ("پایلوت", "گزارش وضعیت"), ("بهره‌برداری", "Runbook")]
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    w, h, xs = 2.0, 1.2, [7.6, 5.15, 2.7, 0.25]
    pos = [(xs[i], 3.3) for i in range(4)] + [(xs[3 - i], 0.6) for i in range(4)]
    cols = ["#eeeeee", "#d6e4f0", "#d6e4f0", "#d6e4f0",
            "#d9ead3", "#f9e3c2", "#f9e3c2", "#f4cccc"]
    for (t, s), (x, y), c in zip(st, pos, cols):
        box(ax, x, y, w, h, t + "\n" + s, c, 10)
    for i in range(3):                        # top row, right to left
        arrow(ax, (pos[i][0] - .08, 3.9), (pos[i + 1][0] + w + .08, 3.9))
    arrow(ax, (1.25, 3.22), (1.25, 1.88))     # down on the left side
    for i in range(4, 7):                     # bottom row, left to right
        arrow(ax, (pos[i][0] + w + .08, 1.2), (pos[i + 1][0] - .08, 1.2))
    # feedback: operations back to project definition (dashed)
    ax.annotate("", xy=(6.15, 3.22), xytext=(8.6, 1.88),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, ls="--", color="#a33",
                                connectionstyle="arc3,rad=0.25"))
    ax.text(8.95, 2.55, fa("بازخورد برای پروژه‌ی بعدی"), ha="center",
            fontsize=8.5, color="#a33")
    save(fig, "appD", "lifecycle.png")


def gate():
    rows = [("همه‌ی آزمون‌های امنیت قبول شده‌اند؟", "نه", "رد، با گزارش", "#f4cccc"),
            ("معیارهای سخت بالای آستانه‌اند؟", "نه", "رد، با گزارش", "#f4cccc"),
            ("بهبود معنادار نسبت به نسخه‌ی فعلی هست؟", "نه", "نسخه‌ی فعلی می‌ماند",
             "#eeeeee"),
            ("کارهای نگهبان افت کرده‌اند؟", "بله", "رد، با گزارش", "#f4cccc")]
    fig, ax = plt.subplots(figsize=(8, 6.4)); ax.axis("off")
    n = len(rows); ax.set_xlim(0, 10); ax.set_ylim(-2.6, n * 1.2 + 1.1)
    top = n * 1.2 + .3
    box(ax, 4.9, top, 4.5, .6, "نسخه‌ی کاندید (مدل، Adapter یا Prompt)", "#d6e4f0")
    arrow(ax, (7.15, top - .02), (7.15, top - .4))
    for i, (q, bad, out, col) in enumerate(rows):
        y = (n - 1 - i) * 1.2 + .3
        box(ax, 4.4, y, 5.5, .75, q, "#f8f8f8")
        box(ax, 0.2, y, 3.3, .75, out, col, 9.5)
        arrow(ax, (4.35, y + .37), (3.55, y + .37))
        ax.text(3.95, y + .5, fa(bad), ha="center", fontsize=8.5)
        good = "نه" if bad == "بله" else "بله"
        arrow(ax, (7.15, y - .02), (7.15, y - .42))
        ax.text(7.4, y - .25, fa(good), ha="center", fontsize=8.5)
    box(ax, 4.4, -.9, 5.5, .75, "انتشار Canary، با درصد کمی از ترافیک", "#d9ead3")
    arrow(ax, (7.15, -.92), (7.15, -1.32))
    box(ax, 4.4, -2.1, 5.5, .75, "انتشار کامل، با مسیر بازگشت تمرین‌شده", "#93c47d")
    ax.set_title(fa("دروازه‌ی انتشار: ترتیب بررسی از بالا به پایین"), fontsize=11)
    save(fig, "appD", "release_gate.png")


if __name__ == "__main__":
    lifecycle()
    gate()
