"""fa_plot.py - matplotlib helper for Persian (RTL) labels with Vazirmatn.
Usage: from fa_plot import fa, setup, save
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]          # /workspace/book
FONT_DIR = ROOT / "assets" / "fonts" / "ttf"
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

def setup():
    for f in FONT_DIR.glob("Vazirmatn*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Vazirmatn", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})

def fa(text: str, digits: bool = False) -> str:
    """Persian label for matplotlib. Matplotlib >= 3.10 (libraqm) shapes and
    reorders RTL text itself, so we only convert digits when asked."""
    if digits:
        text = text.translate(FA_DIGITS)
    return text

def save(fig, chapter: str, name: str) -> Path:
    out = ROOT / "assets" / "img" / chapter
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("saved", p.relative_to(ROOT))
    return p
