"""fa_plot.py - tiny helper: Persian (RTL) labels in matplotlib charts.
Uses the book's Vazirmatn font. Needs matplotlib >= 3.10 (native RTL shaping).
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FONT = ROOT / "assets/fonts/ttf/Vazirmatn-Regular.ttf"
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

def fa(text, digits=True):
    """Return display-ready Persian text (optionally with Persian digits)."""
    s = str(text)
    if digits:
        s = re.sub(r"(?<=\d)\.(?=\d)", "٫", s).translate(FA)
    return s  # matplotlib >= 3.10 shapes RTL text natively (libraqm)

def save(fig, chapter, name):
    out = ROOT / "assets/img" / chapter
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out / name)
