#!/usr/bin/env python3
"""
charts.py — High-resolution Persian diagrams for Chapter 34 (Agentic AI: Skills, Subagents & Multi-Agent Systems).
Generates:
  1. ai_stack_human_body.png: "استک هوش مصنوعی به زبان بدن انسان" (6-card visual infographic inspired by Mohammad's reference)
  2. context_isolation_tokens.png: Context Window Growth across a 50-line BOM (Monolithic Single Agent vs. Progressive Skills + Isolated Subagents)
  3. plan_execute_reflection_trace.png: Plan-and-Execute + Parallel Subagents + Reflection Verifier architecture
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from fa_plot import fa, setup, save


def make_ai_stack_human_body(chapter_folder: str = "ch34"):
    setup()
    fig, ax = plt.subplots(figsize=(12.2, 8.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    # Background canvas
    bg = patches.FancyBboxPatch((0.1, 0.1), 11.8, 8.6, boxstyle="round,pad=0.1,rounding_size=0.25",
                                facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(bg)

    # Header banner
    hdr = patches.FancyBboxPatch((0.4, 7.55), 11.2, 0.95, boxstyle="round,pad=0.08,rounding_size=0.18",
                                 facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2)
    ax.add_patch(hdr)
    ax.text(6.0, 8.15, fa("استک هوش مصنوعی به زبان بدن انسان (Your AI Stack, Explained Like a Human Body)"),
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#F8FAFC")
    ax.text(6.0, 7.77, fa("قبل از ساختن یک سیستم عاملی (Agentic AI)، دقیقاً بدانید هر لایه چه نقشی در معماری شما دارد"),
            ha="center", va="center", fontsize=10.2, color="#38BDF8")

    # 6 Cards in a 2x3 grid (Right-to-Left reading order: row 1 right->left, row 2 right->left)
    cards = [
        # Row 1 (top): 1 (Right), 2 (Middle), 3 (Left)
        (8.15, 4.15, "#DBEAFE", "#2563EB", "#1E40AF",
         "۱. LLM = مغز (The Brain)",
         "مدل زبانی بزرگ",
         "• می‌فهمد، می‌نویسد و استدلال می‌کند\n• فقط چیزهایی را می‌داند که در زمان\n  پیش‌آموزش در وزن‌هایش یاد گرفته است"),
        (4.25, 4.15, "#E0E7FF", "#4F46E5", "#3730A3",
         "۲. RAG = مغز + کتابخانه (Brain + Library)",
         "بازیابی و تولید تقویت‌شده",
         "• قبل از پاسخ، اسناد و مدارک شما را\n  جست‌وجو می‌کند\n• پاسخ‌ها را به داده‌ی واقعی مستند می‌کند"),
        (0.35, 4.15, "#FEF3C7", "#D97706", "#92400E",
         "۳. MCP = سیستم عصبی (Nervous System)",
         "پروتکل استاندارد اتصال ابزار و داده",
         "• یک درگاه استاندارد واحد برای ابزار و داده\n• هوش مصنوعی را امن به نرم‌افزارها،\n  APIها، فایل‌ها و دیتابیس‌ها وصل می‌کند"),
        # Row 2 (bottom): 4 (Right), 5 (Middle), 6 (Left)
        (8.15, 0.55, "#DCFCE7", "#16A34A", "#166534",
         "۴. Skills = حافظه‌ی عضلانی (Muscle Memory)",
         "مهارت‌های ماژولار و دستورالعمل‌های ذخیره‌شده",
         "• دانش و روش کار ذخیره‌شده برای کارهای تکراری\n• افشای تدریجی (Progressive Disclosure):\n  فقط وقتی کار نیاز دارد بارگذاری می‌شود"),
        (4.25, 0.55, "#FCE7F3", "#DB2777", "#9D174D",
         "۵. AI Agent = مغز + دست‌ها (Brain + Hands)",
         "عامل خودمختار در چارچوب سیاست",
         "• مراحل کار را خودش برنامه‌ریزی می‌کند\n• از ابزارها برای اقدام و عمل استفاده می‌کند،\n  نه فقط حرف زدن (همراه با دروازه‌ی انسانی)"),
        (0.35, 0.55, "#F3E8FF", "#9333EA", "#6B21A8",
         "۶. Subagents = تیم متخصصان (Specialists)",
         "ریزعامل‌ها با پنجره‌ی زمینه‌ی ایزوله",
         "• هر متخصص یک زیرمسئله و میز کار تمیز دارد\n• به صورت موازی با کمترین دسترسی اجرا می‌شود\n• فقط چکیده‌ی نهایی را به هماهنگ‌کننده می‌دهد"),
    ]

    card_w = 3.45
    card_h = 3.15
    for x, y, fc, ec, title_col, title, subtitle, bullets in cards:
        rect = patches.FancyBboxPatch((x, y), card_w, card_h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                      facecolor=fc, edgecolor=ec, linewidth=2.0)
        ax.add_patch(rect)
        # Top badge bar inside card
        badge = patches.FancyBboxPatch((x + 0.12, y + card_h - 0.68), card_w - 0.24, 0.54,
                                       boxstyle="round,pad=0.04,rounding_size=0.12",
                                       facecolor=ec, edgecolor="none")
        ax.add_patch(badge)
        ax.text(x + card_w / 2, y + card_h - 0.41, fa(title),
                ha="center", va="center", fontsize=10.3, fontweight="bold", color="#FFFFFF")
        ax.text(x + card_w / 2, y + card_h - 0.98, fa(subtitle),
                ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_col)
        ax.text(x + card_w - 0.18, y + 1.05, fa(bullets),
                ha="right", va="center", fontsize=9.0, color="#0F172A", linespacing=1.65)

    fig.tight_layout()
    save(fig, chapter_folder, "ai_stack_human_body.png")


def make_context_isolation_chart(chapter_folder: str = "ch34"):
    setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    lines = np.array([1, 5, 10, 20, 35, 50])
    # Single agent with monolithic skills (1497) + 866 tokens of raw logs per BOM line
    single_agent_tokens = 1497 + lines * 866
    # Coordinator with progressive skill catalog (150) + 1 loaded SOP (392) + 68 tokens distilled summary per line
    coordinator_tokens = 150 + 392 + lines * 68

    ax1.plot(lines, single_agent_tokens / 1000.0, marker="o", linewidth=2.4, color="#EF4444",
             label=fa("تک‌عامل با پرامپت یکپارچه (بدون ریزعامل)"))
    ax1.plot(lines, coordinator_tokens / 1000.0, marker="s", linewidth=2.4, color="#10B981",
             label=fa("هماهنگ‌کننده + مهارت تدریجی + ریزعامل ایزوله"))
    ax1.axhline(32.0, color="#F59E0B", linestyle="--", linewidth=1.4,
                label=fa("مرز افت دقت توجه (Context Rot ~ ۳۲k)"))

    ax1.set_xlabel(fa("تعداد ردیف‌های BOM در یک درخواست (RFQ)"), fontsize=9.5)
    ax1.set_ylabel(fa("توکن‌های انباشته‌شده در پنجره‌ی متن اصلی (هزار توکن)"), fontsize=9.5)
    ax1.set_title(fa("الف) رشد مصرف پنجره‌ی متن با افزایش ردیف‌های BOM"), fontsize=10.5, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Annotate 50-line point
    ax1.annotate(fa("۴۴٫۸ هزار توکن\n(آلودگی زمینه)", digits=True),
                 xy=(50, 44.8), xytext=(36, 38.5),
                 arrowprops=dict(arrowstyle="->", color="#B91C1C"),
                 fontsize=8.8, fontweight="bold", color="#B91C1C")
    ax1.annotate(fa("۳٫۹۵ هزار توکن\n(۹۱٫۲٪ صرفه‌جویی)", digits=True),
                 xy=(50, 3.95), xytext=(33, 11.5),
                 arrowprops=dict(arrowstyle="->", color="#047857"),
                 fontsize=8.8, fontweight="bold", color="#047857")

    # Right subplot: Breakdown of System Prompt tokens (Monolithic vs Progressive Disclosure)
    labels = [
        fa("پرامپت یکپارچه\n(همه‌ی ۵ مهارت همیشه در پرامپت)"),
        fa("افشای تدریجی (فهرست + ۱ مهارت)\nدر کارهای واقعی"),
    ]
    cat_vals = [1497, 150]
    loaded_vals = [0, 392]

    x = np.arange(len(labels))
    w = 0.42
    ax2.bar(x, cat_vals, width=w, color="#3B82F6", edgecolor="#1E293B", label=fa("توکن پایه در هر نوبت (فهرست/پرامپت)"))
    ax2.bar(x, loaded_vals, width=w, bottom=cat_vals, color="#10B981", edgecolor="#1E293B", label=fa("مهارت بارگذاری‌شده در لحظه (SKILL.md)"))

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel(fa("تعداد توکن‌های دستورالعمل (SOP)"), fontsize=9.5)
    ax2.set_title(fa("ب) اثر افشای تدریجی مهارت‌ها (Progressive Disclosure)"), fontsize=10.5, fontweight="bold", pad=10)
    ax2.set_ylim(0, 1800)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.8, frameon=True)

    ax2.text(0, 1530, fa("۱٬۴۹۷ توکن (همیشه)", digits=True), ha="center", va="bottom", fontweight="bold", color="#1D4ED8")
    ax2.text(1, 575, fa("۵۴۲ توکن (۶۳٫۸٪ کمتر)", digits=True), ha="center", va="bottom", fontweight="bold", color="#047857")

    fig.tight_layout()
    save(fig, chapter_folder, "context_isolation_tokens.png")


def make_plan_execute_reflection_chart(chapter_folder: str = "ch34"):
    setup()
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    nodes = [
        (0.3, 2.1, 2.0, 1.5, "#DBEAFE", "#2563EB",
         "۱. برنامه‌ریز (Planner)\nتجزیه‌ی BOM چهارردیفه\n+ بارگذاری مهارت\nbom-shortage-resolution"),
        (2.8, 3.8, 2.5, 1.35, "#DCFCE7", "#16A34A",
         "ریزعامل ۱: متخصص فنی BOM\nابزار: lookup_part, find_alternates\nپنجره‌ی ایزوله (بدون حق قیمت)"),
        (2.8, 2.15, 2.5, 1.35, "#FEF3C7", "#D97706",
         "ریزعامل ۲: متخصص قیمت‌گذاری\nابزار: predict_price (P10/P50/P90)\nپنجره‌ی ایزوله (فقط خواندنی)"),
        (2.8, 0.5, 2.5, 1.35, "#F3E8FF", "#9333EA",
         "ریزعامل ۳: ریسک زنجیره‌ی تأمین\nابزار: get_lead_time_risk\nپنجره‌ی ایزوله (بررسی گمرک/تأخیر)"),
        (5.9, 2.1, 2.2, 1.5, "#FCE7F3", "#DB2777",
         "۳. بازتاب و ممیزی (Reflection)\nکشف خطای قیمت زیر کف\nدر LM2596S-ADJ (۱٫۰۵۹ < ۱٫۰۹۲)\nو اصلاح خودکار به ۱٫۱۵۰ یورو"),
        (8.6, 2.1, 2.1, 1.5, "#E0F2FE", "#0284C7",
         "۴. دروازه‌ی تأیید انسانی\nپیش‌نویس تمیز و مستند\nدر صف سارا (کارشناس فروش)\nبا ۹۱٪ مصرف توکن کمتر"),
    ]

    for x, y, w, h, fc, ec, txt in nodes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.16",
                                      facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, fa(txt), ha="center", va="center",
                fontsize=8.8, color="#0F172A", linespacing=1.55)

    arrow_kw = dict(arrowstyle="->,head_width=0.25,head_length=0.25", color="#334155", lw=1.7)
    # Planner to 3 subagents
    ax.annotate("", xy=(2.78, 4.45), xytext=(2.32, 3.1), arrowprops=arrow_kw)
    ax.annotate("", xy=(2.78, 2.85), xytext=(2.32, 2.85), arrowprops=arrow_kw)
    ax.annotate("", xy=(2.78, 1.2), xytext=(2.32, 2.6), arrowprops=arrow_kw)
    # 3 subagents to Reflection
    ax.annotate("", xy=(5.88, 3.1), xytext=(5.32, 4.45), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.88, 2.85), xytext=(5.32, 2.85), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.88, 2.6), xytext=(5.32, 1.2), arrowprops=arrow_kw)
    # Reflection to Human gate
    ax.annotate("", xy=(8.58, 2.85), xytext=(8.12, 2.85), arrowprops=arrow_kw)

    ax.set_title(fa("الگوی برنامه‌ریزی-اجرا (Plan-and-Execute)، ریزعامل‌های موازی و حلقه‌ی بازتاب (Reflection)"),
                 fontsize=11.5, fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, chapter_folder, "plan_execute_reflection_trace.png")




def make_web_research_memory_tiers_chart(chapter_folder: str = "ch34"):
    """
    شکل ۳۴-۵: معماری ایجنت جست‌وجوی وب امن (Zero-Leakage Web Research)
    و چرخه‌ی ماندگاری دانش در ۴ لایه‌ی حافظه (4-Tier Agent Memory Persistence).
    """
    from matplotlib.patches import FancyBboxPatch
    setup()
    fig, ax = plt.subplots(figsize=(13.5, 8.8))
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # عنوان اصلی
    ax.text(
        50, 96.2,
        fa("معماری ایجنت جست‌وجوی وب امن و ماندگاری دانش در ۴ لایه‌ی حافظه (4-Tier Memory)"),
        ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A",
    )
    ax.text(
        50, 92.4,
        fa("از تشخیص شکاف دانش و پاک‌سازی کوئری برون‌مرزی تا راستی‌آزمایی متقاطع و ذخیره‌سازی سلسله‌مراتبی در ERP"),
        ha="center", va="center", fontsize=9.8, color="#475569",
    )

    # ردیف بالا: چرخه‌ی ۴ مرحله‌ای تحقیق وب امن (از راست به چپ برای فارسی)
    top_steps = [
        (76.5, 75.0, 20.5, 13.0, "#1E3A8A", "#EFF6FF", "۱. تشخیص شکاف دانش", "قطعه یا PCN در کاتالوگ و وزن‌های\nLLM نیست (STM32N657X0H6Q)"),
        (52.0, 75.0, 20.5, 13.0, "#0F766E", "#F0FDFA", "۲. دروازه پاک‌سازی برون‌مرزی", "حذف نام مشتری (NordSensor)،\nکد RFQ و قیمت هدف پیش از خروج"),
        (27.5, 75.0, 20.5, 13.0, "#B45309", "#FFFBEB", "۳. جست‌وجوی وب در Sandbox", "واکشی دیتاشیت و PCN از منابع\nمجاز (st.com / mouser.com)"),
        (3.0,  75.0, 20.5, 13.0, "#7C3AED", "#F5F3FF", "۴. راستی‌آزمایی و فیلتر تزریق", "تطبیق >= ۲ منبع مستقل و مسدودسازی\nPrompt Injection (Trust=0.96)"),
    ]

    for x, y, w, h, edge_c, fill_c, title_fa, desc_fa in top_steps:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.4",
            facecolor=fill_c, edgecolor=edge_c, linewidth=2.0,
        )
        ax.add_patch(box)
        hdr = FancyBboxPatch(
            (x, y + h - 3.6), w, 3.6,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            facecolor=edge_c, edgecolor=edge_c, linewidth=1.0,
        )
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - 1.8, fa(title_fa), ha="center", va="center", fontsize=9.8, fontweight="bold", color="white")
        ax.text(x + w / 2, y + (h - 3.6) / 2, fa(desc_fa), ha="center", va="center", fontsize=8.5, color="#1E293B", linespacing=1.35)

    # فلش‌های اتصال ردیف بالا (راست به چپ)
    for x_start, x_end in [(76.3, 72.7), (51.8, 48.2), (27.3, 23.7)]:
        ax.annotate(
            "", xy=(x_end, 81.5), xytext=(x_start, 81.5),
            arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2.0, mutation_scale=15),
        )

    # نوار میانی: توزیع‌کننده‌ی حافظه‌ی چهارلایه (Memory Router)
    router_box = FancyBboxPatch(
        (8.0, 61.5), 84.0, 7.2,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor="#0F172A", edgecolor="#334155", linewidth=1.8,
    )
    ax.add_patch(router_box)
    ax.text(
        50, 65.1,
        fa("موتور توزیع و ماندگاری دانش تأییدشده (Multi-Tier Memory Persistence Router) — کجا و با چه ماندگاری ذخیره شود؟"),
        ha="center", va="center", fontsize=10.2, fontweight="bold", color="#F8FAFC",
    )

    # فلش از مرحله‌ی ۴ به نوار میانی
    ax.annotate(
        "", xy=(25.0, 68.9), xytext=(13.2, 74.8),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2.2, mutation_scale=15),
    )

    # چهار لایه‌ی حافظه در ردیف پایین (از راست به چپ: لایه ۱ تا لایه ۴)
    tiers = [
        (
            75.5, 12.0, 21.5, 43.0, "#0284C7", "#E0F2FE",
            "لایهٔ ۱: حافظهٔ کاری و کوتاه‌مدت",
            "Tier 1: Working / Session Memory",
            [
                "• محل ذخیره: Context + Redis",
                "• ماندگاری (TTL): ۳۰ دقیقه (نشست)",
                "• تأخیر بازیابی: < ۵ میلی‌ثانیه",
                "• کلید: session:rfq:STM32N6",
                "• کاربرد: پاسخ فوری به RFQ جاری",
                "  بدون نیاز به فراخوانی مجدد وب",
            ],
        ),
        (
            51.0, 12.0, 21.5, 43.0, "#0D9488", "#CCFBF1",
            "لایهٔ ۲: پروندهٔ تحقیقاتی اپیزودیک",
            "Tier 2: Markdown / JSON Dossier",
            [
                "• محل ذخیره: دیسک / Object Store",
                "• ماندگاری: دائمی (نسخه‌بندی‌شده)",
                "• فایل: STM32N6_dossier.md",
                "• محتوا: خلاصه فنی + URL منابع",
                "  + زمان واکشی + هش SHA-256",
                "• کاربرد: ممیزی انسانی و بازخوانی",
            ],
        ),
        (
            26.5, 12.0, 21.5, 43.0, "#D97706", "#FEF3C7",
            "لایهٔ ۳: حافظهٔ معنایی بلندمدت RAG",
            "Tier 3: Vector / Hybrid Index",
            [
                "• محل ذخیره: Qdrant / Elasticsearch",
                "• ماندگاری (TTL): ۳۰ تا ۹۰ روز",
                "• متادیتا: source_url, trust=0.96,",
                "  fetched_at, injection_scanned",
                "• کاربرد: بازیابی معنایی در تمام",
                "  پروژه‌ها و نشست‌های آینده شرکت",
            ],
        ),
        (
            2.0, 12.0, 21.5, 43.0, "#E11D48", "#FFE4E6",
            "لایهٔ ۴: حافظهٔ ساختاریافتهٔ عملیاتی",
            "Tier 4: SQL ERP Staging Table",
            [
                "• محل ذخیره: MariaDB / ClickHouse",
                "• جدول: part_web_enrichment",
                "• ستون‌ها: lifecycle, lead_weeks,",
                "  benchmark_eur, trust_score",
                "• گارد امنیتی: Staging + تأیید مهندس",
                "  پیش از ارتقا به کاتالوگ اصلی ERP",
            ],
        ),
    ]

    for x, y, w, h, edge_c, fill_c, title_fa, sub_en, bullets in tiers:
        card = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor="white", edgecolor=edge_c, linewidth=2.2,
        )
        ax.add_patch(card)
        top_band = FancyBboxPatch(
            (x, y + h - 8.5), w, 8.5,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            facecolor=edge_c, edgecolor=edge_c, linewidth=1.0,
        )
        ax.add_patch(top_band)
        ax.text(x + w / 2, y + h - 3.2, fa(title_fa), ha="center", va="center", fontsize=9.6, fontweight="bold", color="white")
        ax.text(x + w / 2, y + h - 6.6, sub_en, ha="center", va="center", fontsize=7.8, fontweight="bold", color="#F1F5F9")

        # فلش از نوار میانی به هر لایه
        ax.annotate(
            "", xy=(x + w / 2, y + h + 0.3), xytext=(x + w / 2, 61.3),
            arrowprops=dict(arrowstyle="-|>", color=edge_c, lw=2.2, mutation_scale=15),
        )

        # بالت‌های داخل هر کارت
        curr_y = y + h - 12.2
        for b in bullets:
            ax.text(x + w - 1.2, curr_y, fa(b, digits=False), ha="right", va="center", fontsize=8.3, color="#1E293B")
            curr_y -= 4.8

    # نوار پایینی: سپر دفاعی در برابر مسمومیت حافظه (Memory Poisoning Guard)
    footer = FancyBboxPatch(
        (2.0, 2.2), 95.0, 6.8,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.8,
    )
    ax.add_patch(footer)
    ax.text(
        50, 5.6,
        fa("سپر دفاعی مسمومیت حافظه (Memory Poisoning Guard): هیچ داده‌ی وبی بدون اسکن Prompt Injection، امتیاز اعتماد >= 0.70 و تطبیق >= ۲ منبع وارد لایه‌های ۲ تا ۴ نمی‌شود."),
        ha="center", va="center", fontsize=9.2, fontweight="bold", color="#065F46",
    )

    save(fig, chapter_folder, "web_research_memory_tiers.png")


if __name__ == "__main__":
    ch = sys.argv[1] if len(sys.argv) > 1 else "ch34"
    make_ai_stack_human_body(ch)
    make_context_isolation_chart(ch)
    make_plan_execute_reflection_chart(ch)
    make_web_research_memory_tiers_chart(ch)
