"""charts.py - figures for chapter 10 (all data synthetic / from the worked examples).
Writes SVG files into assets/img/ch10/.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics import precision_recall

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch10"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})
BLUE, ORANGE, GREY, RED = "#1f5f8b", "#e08a1e", "#9aa5b1", "#c0392b"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg")
    plt.close(fig)
    print("wrote", OUT / name)


# 1) six predictions: absolute vs squared errors
parts = ["NE555DR", "ATMEGA328P", "LM2596S", "TPS5430", "BC847C", "STM32F407"]
y = np.array([1.00, 1.20, 0.80, 2.50, 0.50, 1.50])
yhat = np.array([1.10, 1.00, 0.90, 2.00, 0.60, 1.60])
err = np.abs(yhat - y)
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
ax[0].bar(parts, err, color=BLUE)
ax[0].axhline(err.mean(), color=ORANGE, ls="--", label=f"MAE = {err.mean():.3f}")
ax[0].set_title("|error| per row (EUR)")
ax[0].legend(frameon=False)
share = err ** 2 / np.sum(err ** 2)
ax[1].bar(parts, share * 100, color=[RED if s > .5 else GREY for s in share])
ax[1].set_title("share of squared error (%)")
for a in ax:
    a.tick_params(axis="x", rotation=60, labelsize=8)
save(fig, "errors_six_rows.svg")

# 2) predicted vs actual with 45-degree line
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.plot([0.3, 2.7], [0.3, 2.7], color=GREY, lw=1, label="perfect")
ax.scatter(y, yhat, color=BLUE, zorder=3)
for p, a, b in zip(parts, y, yhat):
    ax.annotate(p, (a, b), textcoords="offset points", xytext=(4, -10), fontsize=7)
ax.set_xlabel("actual price y (EUR)")
ax.set_ylabel("forecast y_hat (EUR)")
ax.set_title("below the line = under-forecast")
ax.legend(frameon=False, loc="upper left")
save(fig, "pred_vs_actual.svg")

# 3) pinball loss shape for tau = 0.1, 0.5, 0.9
d = np.linspace(-1, 1, 201)          # y - q
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for tau, c in [(0.1, GREY), (0.5, BLUE), (0.9, ORANGE)]:
    ax.plot(d, np.maximum(tau * d, (tau - 1) * d), color=c, label=f"tau = {tau}")
ax.set_xlabel("y - q   (actual minus predicted quantile)")
ax.set_ylabel("loss")
ax.legend(frameon=False)
ax.set_title("pinball loss: P90 punishes 'actual above q' 9x more")
save(fig, "pinball_shape.svg")

# 4) precision / recall vs threshold, eight quotes
won = [1, 1, 0, 1, 0, 1, 0, 0]
p = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
th = np.linspace(0.05, 0.95, 91)
pr = np.array([precision_recall(won, p, t) for t in th])
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.step(th, pr[:, 0], where="post", color=BLUE, label="precision")
ax.step(th, pr[:, 1], where="post", color=ORANGE, label="recall")
ax.axvline(0.5, color=GREY, ls=":")
ax.set_xlabel("threshold")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False)
ax.set_title("raising the threshold: precision up, recall down")
save(fig, "precision_recall_threshold.svg")

# 5) ROC and PR curves for the eight quotes
order = np.argsort(-np.array(p))
t = np.array(won)[order]
tpr = np.concatenate([[0], np.cumsum(t) / t.sum()])
fpr = np.concatenate([[0], np.cumsum(1 - t) / (1 - t).sum()])
prec = np.cumsum(t) / np.arange(1, len(t) + 1)
rec = np.cumsum(t) / t.sum()
fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.0))
ax[0].plot(fpr, tpr, marker="o", color=BLUE)
ax[0].fill_between(fpr, tpr, step=None, alpha=.15, color=BLUE)
ax[0].plot([0, 1], [0, 1], color=GREY, ls="--")
ax[0].set_title("ROC (AUC = 0.81)")
ax[0].set_xlabel("false positive rate")
ax[0].set_ylabel("recall (TPR)")
ax[1].plot(rec, prec, marker="o", color=ORANGE)
ax[1].axhline(0.5, color=GREY, ls="--", label="random = base rate 0.5")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("precision-recall (AP = 0.85)")
ax[1].set_xlabel("recall")
ax[1].set_ylabel("precision")
ax[1].legend(frameon=False, fontsize=8, loc="lower left")
save(fig, "roc_pr_eight_quotes.svg")

# 6) reliability diagram from the calibration table (example numbers)
mean_pred = [0.12, 0.30, 0.50, 0.70, 0.90]
observed = [0.15, 0.30, 0.40, 0.55, 0.70]
n = [80, 100, 100, 80, 40]
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.plot([0, 1], [0, 1], color=GREY, ls="--", label="perfectly calibrated")
ax.scatter(mean_pred, observed, s=[k * 2 for k in n], color=BLUE, zorder=3,
           label="bins (size = n)")
ax.plot(mean_pred, observed, color=BLUE)
for mp, ob in zip(mean_pred, observed):
    ax.vlines(mp, ob, mp, color=RED, lw=1)
ax.set_xlabel("mean predicted win probability")
ax.set_ylabel("observed win rate")
ax.set_title("over-confident at the top")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "reliability_diagram.svg")

# 7) segment WAPE from report.py (synthetic run, numbers copied from its output)
segs = ["ALL", "ST", "TI", "Murata", "Microchip", "qty100", "qty1000", "qty20000"]
wb = [0.142, 0.142, 0.141, 0.147, 0.137, 0.137, 0.140, 0.140]
wc = [0.096, 0.083, 0.082, 0.083, 0.201, 0.095, 0.099, 0.094]
x = np.arange(len(segs))
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.bar(x - 0.2, wb, 0.4, color=GREY, label="baseline")
ax.bar(x + 0.2, wc, 0.4, color=[RED if c > b else BLUE for c, b in zip(wc, wb)],
       label="candidate")
ax.set_xticks(x, segs, rotation=30, fontsize=8)
ax.set_ylabel("WAPE")
ax.legend(frameon=False)
ax.set_title("the average hides one segment where the model is worse")
save(fig, "segment_wape.svg")

# 8) sample size needed vs detectable effect
delta = np.linspace(0.003, 0.05, 200)
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for sigma, c in [(0.05, GREY), (0.10, BLUE), (0.20, ORANGE)]:
    ax.plot(delta, (2 * sigma / delta) ** 2, color=c, label=f"sigma = {sigma} EUR")
ax.set_yscale("log")
ax.set_xlabel("improvement in MAE you want to detect (EUR)")
ax.set_ylabel("independent rows needed")
ax.scatter([0.01], [400], color=RED, zorder=3)
ax.annotate("sigma 0.10, delta 0.01 -> 400", (0.01, 400), xytext=(0.018, 1500),
            arrowprops=dict(arrowstyle="->", color=RED), fontsize=8)
ax.legend(frameon=False)
save(fig, "sample_size.svg")
