"""charts.py - figures for chapter 11 (synthetic data). Writes SVGs to assets/img/ch11/."""
import json
import math
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets/img/ch11"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREY, RED, GREEN = "#1f5f8b", "#e08a1e", "#9aa5b1", "#c0392b", "#2e8b57"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg")
    plt.close(fig)
    print("wrote", name)


# 1) activation functions
z = np.linspace(-4, 4, 400)
gelu = 0.5 * z * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
fig, ax = plt.subplots(figsize=(5.4, 2.9))
ax.plot(z, 1 / (1 + np.exp(-z)), color=GREY, label="sigmoid")
ax.plot(z, np.tanh(z), color=GREEN, label="tanh")
ax.plot(z, np.maximum(0, z), color=BLUE, lw=2, label="ReLU")
ax.plot(z, gelu, color=ORANGE, ls="--", lw=2, label="GELU")
ax.axhline(0, color="k", lw=.5)
ax.axvline(0, color="k", lw=.5)
ax.set_ylim(-1.3, 4)
ax.set_xlabel("z = w.x + b")
ax.legend(frameon=False)
save(fig, "activations.svg")

# 2) gradient descent on 1-D loss L(w) = (w - 2)^2 with three learning rates
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
ws = np.linspace(-1.5, 5.5, 200)
for ax, lr, title in zip(axes, [0.05, 0.4, 1.05], ["lr = 0.05 (slow)", "lr = 0.4 (good)",
                                                    "lr = 1.05 (diverges)"]):
    ax.plot(ws, (ws - 2) ** 2, color=GREY)
    w, path = -1.0, []
    for _ in range(8):
        path.append(w)
        w = w - lr * 2 * (w - 2)
    path = np.array(path)
    ax.plot(path, (path - 2) ** 2, marker="o", color=RED if lr > 1 else BLUE, ms=4)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(0, 14)
    ax.set_xlabel("w")
axes[0].set_ylabel("loss")
save(fig, "learning_rates.svg")

# 3) loss curves from torch_mlp.py (real run history)
h = json.load(open(pathlib.Path(__file__).parent / "torch_mlp_history.json"))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for name, c in [("linear", GREY), ("mlp_dropout", ORANGE), ("mlp", BLUE)]:
    ax.plot(h[name]["train"], color=c, lw=1, ls=":")
    ax.plot(h[name]["val"], color=c, lw=2, label=f"{name} (val)")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE on log-price")
ax.set_title("dotted = train, solid = validation")
ax.legend(frameon=False, fontsize=8)
save(fig, "loss_curves.svg")

# 4) overfitting: 20 noisy points, big MLP with and without weight decay
torch.manual_seed(0)
rng = np.random.default_rng(3)
x = np.sort(rng.uniform(-1, 1, 20)).astype(np.float32)
y = (1.2 - 0.25 * x + 0.08 * x * x + rng.normal(0, 0.05, 20)).astype(np.float32)
xv = np.linspace(-1, 1, 200).astype(np.float32)
yv = (1.2 - 0.25 * xv + 0.08 * xv * xv).astype(np.float32)
X, Y = torch.tensor(x)[:, None], torch.tensor(y)[:, None]
XV, YV = torch.tensor(xv)[:, None], torch.tensor(yv)[:, None]
curves, fits = {}, {}
for wd in (0.0, 1e-2):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    tr, va = [], []
    for ep in range(3000):
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            tr.append(loss.item())
            va.append(((net(XV) - YV) ** 2).mean().item())
    curves[wd] = (tr, va)
    with torch.no_grad():
        fits[wd] = net(XV).squeeze().numpy()
    print(f"wd={wd}: final train={tr[-1]:.5f} val={va[-1]:.5f} best val={min(va):.5f} "
          f"at epoch {int(np.argmin(va))}")
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
ax[0].plot(curves[0.0][0], color=BLUE, ls=":", label="train")
ax[0].plot(curves[0.0][1], color=RED, label="validation")
ax[0].set_yscale("log")
ax[0].set_xlabel("epoch")
ax[0].set_title("no weight decay", fontsize=9)
ax[0].legend(frameon=False, fontsize=8)
ax[1].scatter(x, y, color="k", s=12, label="20 training points")
ax[1].plot(xv, yv, color=GREY, lw=3, label="true curve")
ax[1].plot(xv, fits[0.0], color=RED, lw=1, label="big MLP")
ax[1].set_xlabel("x = log10(qty) - 3")
ax[1].set_ylabel("price (EUR)")
ax[1].legend(frameon=False, fontsize=7)
save(fig, "overfitting.svg")

# 5) micrograd toy: fitted curve vs points (re-run tiny training here)
import random
from micrograd import MLP, Value
random.seed(0)
data = [(q, 1.2 - 0.25 * q + 0.08 * q * q) for q in [-1, -0.5, 0, 0.5, 1, 1.3]]
net = MLP([1, 8, 1])
snaps = {}
for epoch in range(301):
    loss = sum(((net([a]) - b) ** 2 for a, b in data), Value(0.0)) * (1 / len(data))
    for p in net.parameters():
        p.grad = 0.0
    loss.backward()
    for p in net.parameters():
        p.data -= 0.05 * p.grad
    if epoch in (0, 20, 300):
        snaps[epoch] = [net([g]).data for g in np.linspace(-1.2, 1.5, 60)]
fig, ax = plt.subplots(figsize=(4.8, 2.9))
g = np.linspace(-1.2, 1.5, 60)
for (ep, ys), c in zip(snaps.items(), [GREY, ORANGE, BLUE]):
    ax.plot(g, ys, color=c, label=f"after epoch {ep}")
ax.scatter([d[0] for d in data], [d[1] for d in data], color="k", zorder=3, label="data")
ax.set_xlabel("x = log10(qty) - 3")
ax.set_ylabel("price (EUR)")
ax.legend(frameon=False, fontsize=8)
save(fig, "micrograd_fit.svg")
