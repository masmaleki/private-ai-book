#!/usr/bin/env python3
"""
charts.py — High-resolution Persian diagrams for the NLP chapter.
Generates:
  1. nlp_generations_tradeoff.png: Latency (ms, log scale) vs. Flexibility/Accuracy across the 4 NLP Generations
  2. hybrid_triage_funnel.png: Hybrid NLP + LLM Pipeline Funnel on 1,000 incoming Alpenline emails
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from fa_plot import fa, setup, save


def make_generations_chart(chapter_folder: str = "ch12"):
    setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    stages = [
        fa("نسل ۱: قواعد و Regex\n(نرمال‌سازی و الگو)"),
        fa("نسل ۲: آماری\n(TF-IDF / BM25 / NB)"),
        fa("نسل ۳: بردار و Subword\n(Word2Vec / FastText)"),
        fa("نسل ۴ الف: Encoder\n(BERT / ParsBERT)"),
        fa("نسل ۴ ب: Decoder LLM\n(Qwen / Llama 8B-70B)"),
    ]
    latencies_ms = [0.08, 0.45, 1.8, 8.5, 850.0]
    colors = ["#0EA5E9", "#10B981", "#6366F1", "#F59E0B", "#EF4444"]

    y_pos = np.arange(len(stages))
    bars = ax1.barh(y_pos, latencies_ms, color=colors, height=0.58, edgecolor="#1E293B", linewidth=0.8)
    ax1.set_xscale("log")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stages, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel(fa("تأخیر پردازش هر ایمیل بر حسب میلی‌ثانیه (مقیاس لگاریتمی)"), fontsize=9.5)
    ax1.set_title(fa("الف) مقایسه‌ی سرعت ۵ رویکرد پردازش زبان طبیعی"), fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlim(0.03, 3500)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, val in zip(bars, latencies_ms):
        label = fa(f" {val} ms", digits=True)
        ax1.text(val * 1.18, bar.get_y() + bar.get_height() / 2, label,
                 va="center", ha="left", fontsize=8.8, fontweight="bold", color="#0F172A")

    # Right panel: Hybrid vs LLM-only average latency and GPU cost on 1000 emails
    categories = [
        fa("فقط LLM بزرگ\n(۱۰۰٪ ایمیل‌ها در GPU)"),
        fa("پایپ‌لاین ترکیبی ما\n(۷۸٪ کلاسیک/Encoder + ۲۲٪ LLM)"),
    ]
    avg_lat = [850.0, 187.4]
    gpu_pct = [100.0, 22.0]

    x = np.arange(len(categories))
    w = 0.34
    b1 = ax2.bar(x - w/2, avg_lat, width=w, color="#3B82F6", edgecolor="#1E293B", label=fa("میانگین تأخیر (ms)"))
    b2 = ax2.bar(x + w/2, [g * 8.5 for g in gpu_pct], width=w, color="#10B981", edgecolor="#1E293B", label=fa("مصرف نسبی GPU (مقیاس‌شده)"))

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9.5)
    ax2.set_ylabel(fa("میلی‌ثانیه / شاخص بار محاسباتی"), fontsize=9.5)
    ax2.set_title(fa("ب) اثر پایپ‌لاین ترکیبی (کاهش ۴٫۵ برابری تأخیر و ۷۸٪ صرفه‌جویی GPU)"), fontsize=10.5, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, v in zip(b1, avg_lat):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 18,
                 fa(f"{v} ms", digits=True), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1D4ED8")
    for bar, g in zip(b2, gpu_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 18,
                 fa(f"بار GPU: {int(g)}٪", digits=True), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#047857")

    ax2.set_ylim(0, 1020)
    ax2.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, chapter_folder, "nlp_generations_tradeoff.png")


def make_funnel_chart(chapter_folder: str = "ch12"):
    setup()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.4, 3.8, 2.0, 1.5, "#E0F2FE", "#0284C7",
         "۱. ورودی خام (۱۰۰۰ ایمیل)\nنرمال‌سازی فارسی و فنی\n(نیم‌فاصله، ی/ک، اعداد، MPN)\nتأخیر: ۰٫۰۸ ms"),
        (2.8, 3.8, 2.1, 1.5, "#DCFCE7", "#16A34A",
         "۲. استخراج قطعی (Regex/NER)\nو طبقه‌بندی سریع (NB/BERT)\nتشخیص نیت، فوریت و MPN\nتأخیر: ۰٫۴۵ تا ۸ ms"),
        (5.4, 4.2, 4.1, 1.35, "#D1FAE5", "#059669",
         "مسیر سریع (۷۸٪ = ۷۸۰ ایمیل)\nدرخواست‌های استاندارد و شفاف (اعتماد ≥ ۰٫۷۵)\nپاسخ/ثبت خودکار در صف بدون بیدار کردن LLM"),
        (5.4, 2.1, 4.1, 1.5, "#FEF3C7", "#D97706",
         "مسیر استدلال عمیق (۲۲٪ = ۲۲۰ ایمیل)\nایمیل‌های مبهم، چندمنظوره یا نیازمند قطعه جایگزین\nارسال به LLM خصوصی همراه با موجودیت‌های استخراج‌شده"),
        (1.5, 0.4, 7.2, 1.15, "#F1F5F9", "#334155",
         "نتیجه در ERP آلپن‌لاین: ۱۰۰٪ پوشش ایمیل‌ها | ۷۸٪ کاهش مصرف GPU | صفر دستکاری خاموش MPN"),
    ]

    for x, y, w, h, fc, ec, text in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.18",
                                      facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, fa(text), ha="center", va="center", fontsize=9.3,
                color="#0F172A", fontweight="medium", linespacing=1.55)

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.28,head_length=0.28", color="#334155", lw=1.8)
    ax.annotate("", xy=(2.78, 4.55), xytext=(2.42, 4.55), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.38, 4.88), xytext=(4.92, 4.65), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.38, 2.85), xytext=(4.92, 4.35), arrowprops=arrow_kw)

    ax.set_title(fa("معماری قیف پردازش زبان طبیعی ترکیبی (Hybrid NLP + LLM Funnel) در ERP"),
                 fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, chapter_folder, "hybrid_triage_funnel.png")


if __name__ == "__main__":
    ch = sys.argv[1] if len(sys.argv) > 1 else "ch12"
    make_generations_chart(ch)
    make_funnel_chart(ch)
