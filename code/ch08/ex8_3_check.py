"""Exercise 8-1 and 8-3: check the hand calculations (example data)."""
import numpy as np

from hand_examples import best_split, sse

qty = np.array([20, 40, 200, 400, 800])
y = np.array([3.0, 2.8, 2.0, 1.9, 1.8])
print(f"8-1 before: SSE={sse(y):.4f}")
for thr, s, lm, rm in best_split(qty, y):
    print(f"    qty < {thr:4.0f}: SSE={s:.4f}")
F, lr = np.full(5, y.mean()), 0.5
print(f"8-3 round 0: MSE={((y - F) ** 2).mean():.4f}")
for rnd in (1, 2):
    r = y - F
    thr, _, lm, rm = min(best_split(qty, r), key=lambda t: t[1])
    F = F + lr * np.where(qty < thr, lm, rm)
    mse = ((y - F) ** 2).mean()
    print(f"    round {rnd}: split < {thr:.0f}  F={F.round(3)}  MSE={mse:.4f}")
