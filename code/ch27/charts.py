"""charts.py - figures for chapter 26 (example data)."""
import sys; sys.path.insert(0, "../common")
import numpy as np
import matplotlib.pyplot as plt
from fa_plot import fa, setup, save
from alternates import CANDIDATES, ORIGINAL, REQ, hard_filter, soft_score, WEIGHTS

setup()
names = {"package": "بسته‌بندی", "pinout": "ترتیب پایه‌ها", "vin": "ولتاژ ورودی",
         "iout": "جریان خروجی", "fsw": "فرکانس", "lifecycle": "چرخه‌ی عمر"}
colors = ["#1f3a5f", "#3d6a99", "#7fa7d1", "#e0a43a", "#c9563c", "#6b9e5a"]
passed = [c for c in CANDIDATES if not hard_filter(c, REQ)]

# Fig 1: stacked weighted contributions
fig, ax = plt.subplots(figsize=(6.2, 3.2))
left = np.zeros(len(passed))
for (k, w), col in zip(WEIGHTS.items(), colors):
    vals = np.array([w * soft_score(c, ORIGINAL)[k] for c in passed])
    ax.barh([c.mpn for c in passed], vals, left=left, color=col,
            label=fa(names[k]))
    left += vals
for i, v in enumerate(left):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1.12)
ax.invert_yaxis()
ax.set_xlabel(fa("امتیاز نرم (سهم هر ویژگی)"))
ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          frameon=False)
save(fig, "ch27", "score_breakdown.png")

# Fig 2: hard-filter check map (pass / fail per requirement)
checks = ["vin", "iout", "temp", "life"]
labels = [fa("ولتاژ حداقل ۳۲"), fa("جریان حداقل ۲٫۵"), fa("دما ۴۰- تا ۱۰۵"), fa("EOL نباشد")]
M = []
for c in CANDIDATES:
    M.append([c.vin_max >= REQ["vin_needed"], c.iout_max >= REQ["iout_needed"],
              c.temp_min <= REQ["temp_min"] and c.temp_max >= REQ["temp_max"],
              c.lifecycle != "EOL"])
M = np.array(M, dtype=float)
fig, ax = plt.subplots(figsize=(5.6, 2.6))
ax.imshow(M, cmap=plt.matplotlib.colors.ListedColormap(["#e8a39a", "#a8d5a2"]),
          aspect="auto")
ax.set_xticks(range(4), labels, fontsize=9)
ax.set_yticks(range(len(CANDIDATES)), [c.mpn for c in CANDIDATES], fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, "✓" if M[i, j] else "✗", ha="center", va="center",
                fontsize=12, family="DejaVu Sans")
ax.set_title(fa("فیلترهای سخت: سبز = قبول، قرمز = رد"), fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
save(fig, "ch27", "hard_filters.png")

# Fig 3: precision@k illustration for 4 example cases (exercise 26-7 data)
fig, ax = plt.subplots(figsize=(5.2, 2.6))
cases = [fa("مورد ۱"), fa("مورد ۲"), fa("مورد ۳"), fa("مورد ۴")]
p3 = [3/3, 1/3, 2/3, 0]
ax.bar(cases, p3, color="#3d6a99")
ax.axhline(np.mean(p3), ls="--", color="#c9563c")
ax.text(3.4, np.mean(p3) + 0.03, fa("میانگین ۰٫۵۰"), ha="right", fontsize=9,
        color="#c9563c")
ax.set_ylim(0, 1.1); ax.set_ylabel("Precision@3")
save(fig, "ch27", "precision_at_k.png")
