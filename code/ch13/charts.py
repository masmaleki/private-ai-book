"""charts.py - figures for chapter 12. Uses bpe.py and real tokenizers (see real_tokenizers.py)."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from bpe import CORPUS, train, encode
from real_tokenizers import pieces, SENT, ALL

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch12"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False})
COLORS = ["#1f5f8b", "#e08a1e", "#2e8b57", "#9b59b6", "#c0392b", "#16a085", "#7f8c8d"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg")
    plt.close(fig)
    print("wrote", name)


# 1) token counts per language and tokenizer (real counts)
langs = list(SENT)
fig, ax = plt.subplots(figsize=(6.0, 3.0))
w = 0.2
for i, name in enumerate(ALL):
    counts = [len(pieces(name, SENT[l])[0]) for l in langs]
    bars = ax.bar([x + (i - 1.5) * w for x in range(3)], counts, w, color=COLORS[i], label=name)
    ax.bar_label(bars, fontsize=7)
ax.set_xticks(range(3), ["English", "German", "Persian"])
ax.set_ylabel("tokens")
ax.set_title("same RFQ sentence, four tokenizers")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
ax.set_ylim(0, 100)
save(fig, "tokens_per_language.svg")

# 2) BPE training: vocab grows, corpus token count shrinks
sizes, lengths = [], []
for k in range(0, 13):
    merges, vocab, words = train(CORPUS, k, verbose=False)
    sizes.append(len(vocab))
    lengths.append(sum(len(wd) * f for wd, f in words.items()))
fig, ax1 = plt.subplots(figsize=(5.4, 2.8))
ax1.plot(range(13), lengths, marker="o", color=COLORS[0], label="tokens in corpus")
ax1.set_xlabel("number of merges")
ax1.set_ylabel("tokens in corpus", color=COLORS[0])
ax2 = ax1.twinx()
ax2.plot(range(13), sizes, marker="s", color=COLORS[1], label="vocabulary size")
ax2.set_ylabel("vocabulary size", color=COLORS[1])
ax2.spines["top"].set_visible(False)
ax1.set_title("each merge: +1 vocab entry, fewer tokens")
save(fig, "bpe_tradeoff.svg")


# 3) token boundaries for one MPN under three tokenizers
def draw_tokens(ax, y, label, toks):
    x = 0.0
    ax.text(-0.3, y + 0.35, label, ha="right", va="center", fontsize=9)
    for i, t in enumerate(toks):
        t = t.replace("\u2581", "_")
        wdt = 0.28 * max(len(t), 1) + 0.12
        ax.add_patch(FancyBboxPatch((x, y), wdt, 0.7, boxstyle="round,pad=0.02",
                                    fc=COLORS[i % len(COLORS)], ec="white", alpha=.85))
        ax.text(x + wdt / 2, y + 0.35, t, ha="center", va="center", color="white",
                fontsize=9, family="monospace")
        x += wdt + 0.05
    ax.text(x + 0.2, y + 0.35, f"{len(toks)} tokens", va="center", fontsize=8)


merges, _, _ = train(CORPUS, 12, verbose=False)
mpn = "STM32F407VGT6"
fig, ax = plt.subplots(figsize=(6.4, 2.4))
draw_tokens(ax, 2, "our toy BPE", encode(mpn, merges))
draw_tokens(ax, 1, "Qwen2.5", pieces("Qwen2.5", mpn)[0])
draw_tokens(ax, 0, "Mistral v0.3", pieces("Mistral-v0.3", mpn)[0])
ax.set_xlim(-2.6, 6.2)
ax.set_ylim(-0.2, 2.9)
ax.axis("off")
ax.set_title(f"how three tokenizers cut {mpn}", fontsize=10)
save(fig, "mpn_boundaries.svg")

# 4) tokens per character by language (efficiency)
fig, ax = plt.subplots(figsize=(5.4, 2.8))
for i, name in enumerate(ALL):
    ratio = [len(pieces(name, SENT[l])[0]) / len(SENT[l]) for l in langs]
    ax.plot(langs, ratio, marker="o", color=COLORS[i], label=name)
ax.set_ylabel("tokens per character")
ax.set_title("lower = cheaper and more text fits in the context")
ax.legend(frameon=False, fontsize=8)
save(fig, "tokens_per_char.svg")
