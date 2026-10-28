"""torch_mlp.py - the same ideas in PyTorch: autograd check + a small price MLP.
All data is synthetic (example data), generated with a fixed seed.
"""
import time
import numpy as np
import torch
from torch import nn

torch.manual_seed(42)

# ---- 1) check the hand-computed 2-2-1 example with autograd ----
W1 = torch.tensor([[0.5, -0.2], [0.3, 0.8]], requires_grad=True)
b1 = torch.tensor([0.1, -0.1], requires_grad=True)
w2 = torch.tensor([0.7, -0.4], requires_grad=True)
b2 = torch.tensor(0.2, requires_grad=True)
x, t = torch.tensor([1.0, 0.5]), torch.tensor(0.5)
h = torch.relu(W1 @ x + b1)
y = w2 @ h + b2
loss = (y - t) ** 2
loss.backward()
print(f"y_hat={y.item():.4f} loss={loss.item():.4f}")
print("dL/dW1 =", W1.grad.tolist())
print("dL/dw2 =", w2.grad.tolist(), " dL/db2 =", round(b2.grad.item(), 4))

# ---- 2) synthetic price dataset ----
rng = np.random.default_rng(42)
N = 4000
log_qty = rng.uniform(2, 4.5, N)                 # log10 of quantity: 100 .. ~30k
lead = rng.uniform(1, 30, N)                     # quoted lead time in weeks
cat = rng.integers(0, 4, N)                      # 4 part categories
base = np.array([0.08, 1.5, 4.0, 0.6])[cat]      # synthetic base price per category
# nonlinear truth with interactions (synthetic): volume discount differs per category,
# scarcity premium appears only when lead time is long, and only for some categories
slope = np.array([-0.05, -0.20, -0.10, -0.30])[cat]
premium = np.array([0.0, 0.8, 0.1, 1.2])[cat]
price = base * 10 ** (slope * (log_qty - 3)) * (1 + premium / (1 + np.exp(-(lead - 18) / 2)))
price *= np.exp(rng.normal(0, 0.05, N))
X = np.column_stack([log_qty, lead / 10, np.eye(4)[cat]]).astype(np.float32)
y = np.log(price).astype(np.float32)             # learn log-price

idx = rng.permutation(N)                         # random split is ok here: no time axis
tr, va = idx[:3000], idx[3000:]
Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])


HISTORY = {}


def fit(model, epochs=150, lr=3e-3, wd=0.0, batch=256, name=None):
    hist = HISTORY.setdefault(name or str(len(HISTORY)), {"train": [], "val": []})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]).squeeze(-1), ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            trl = loss_fn(model(Xtr).squeeze(-1), ytr).item()
            val = loss_fn(model(Xva).squeeze(-1), yva).item()
        hist["train"].append(trl)
        hist["val"].append(val)
        if ep in (0, 9, 49, epochs - 1):
            print(f"  epoch {ep + 1:3d}  train_mse={trl:.5f}  val_mse={val:.5f}")
    return model


def mae_eur(model):
    with torch.no_grad():
        p = torch.exp(model(Xva).squeeze(-1)).numpy()
    return float(np.mean(np.abs(p - np.exp(yva.numpy()))))


print("\nlinear model (no hidden layer):")
lin = fit(nn.Linear(6, 1), name="linear")
print("\nMLP 6-32-32-1 with ReLU, dropout 0.1, weight decay 1e-4:")
mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
t0 = time.time()
fit(mlp, wd=1e-4, name="mlp_dropout")
print(f"  training time on CPU: {time.time() - t0:.1f} s")
print("\nsame MLP without dropout:")
mlp_nd = nn.Sequential(nn.Linear(6, 32), nn.ReLU(),
                       nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
fit(mlp_nd, wd=1e-4, name="mlp")
n_params = sum(p.numel() for p in mlp.parameters())
print(f"\nparameters: linear={sum(p.numel() for p in lin.parameters())}, mlp={n_params}")
print(f"validation MAE in EUR: linear={mae_eur(lin):.4f}  mlp_dropout={mae_eur(mlp):.4f}"
      f"  mlp_no_dropout={mae_eur(mlp_nd):.4f}")

import json
json.dump(HISTORY, open("torch_mlp_history.json", "w"))
