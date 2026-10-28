"""Chapter 8: a regression tree split and two boosting rounds, by hand (example data)."""
import numpy as np

qty = np.array([10, 50, 100, 500, 1000, 5000])
y = np.array([2.00, 1.90, 1.80, 1.20, 1.10, 1.00])


def sse(v):
    return float(((v - v.mean()) ** 2).sum()) if len(v) else 0.0


def best_split(x, r):
    rows = []
    for i in range(1, len(x)):
        thr = (x[i - 1] + x[i]) / 2
        left, right = r[x < thr], r[x >= thr]
        rows.append((thr, sse(left) + sse(right), left.mean(), right.mean()))
    return rows


if __name__ == "__main__":
    print("== 1) one split: sum of squared errors for every threshold ==")
    print(f"before split: SSE = {sse(y):.4f}")
    for thr, s, lm, rm in best_split(qty, y):
        print(f"qty < {thr:6.0f}: SSE = {s:.4f}  left mean = {lm:.3f}  "
              f"right mean = {rm:.3f}")

    print("\n== 2) gradient boosting with stumps, learning rate 0.5 ==")
    lr, F = 0.5, np.full(len(y), y.mean())
    print(f"round 0: F = {F.round(3)}  MSE = {((y - F) ** 2).mean():.4f}")
    for rnd in (1, 2):
        r = y - F  # residuals = negative gradient of MSE/2
        thr, _, lm, rm = min(best_split(qty, r), key=lambda t: t[1])
        step = np.where(qty < thr, lm, rm)
        F = F + lr * step
        print(f"round {rnd}: residuals = {r.round(3)}")
        print(f"         split qty < {thr:.0f}  leaf values = ({lm:+.3f}, {rm:+.3f})")
        print(f"         F = {F.round(3)}  MSE = {((y - F) ** 2).mean():.4f}")

    print("\n== 3) pinball (quantile) loss for q = 0.9, true value 10 ==")
    for pred in (9.0, 11.0):
        e = 10 - pred
        loss = max(0.9 * e, (0.9 - 1) * e)
        print(f"prediction {pred:4.1f}: loss = {loss:.2f}")
