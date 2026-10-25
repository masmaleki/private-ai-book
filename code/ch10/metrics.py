"""metrics.py - evaluation metrics for forecasting and classification (teaching module).
Pure NumPy, no other dependencies. Every function takes plain lists or arrays.
"""
import numpy as np


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


# ---------- point-forecast metrics ----------
def mae(y, yhat):
    y, yhat = _arr(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    y, yhat = _arr(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat):
    """Mean absolute percentage error. Undefined when any y == 0."""
    y, yhat = _arr(y, yhat)
    if np.any(y == 0):
        raise ValueError("MAPE undefined: actual value 0 present")
    return float(np.mean(np.abs(y - yhat) / np.abs(y)))


def smape(y, yhat):
    """Symmetric MAPE, version with (|y|+|yhat|)/2 in the denominator."""
    y, yhat = _arr(y, yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    return float(np.mean(np.abs(y - yhat) / denom))


def wape(y, yhat, weights=None):
    """Weighted APE = sum|err| / sum|y|. Optional weights (e.g. quantity)."""
    y, yhat = _arr(y, yhat)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    return float(np.sum(w * np.abs(y - yhat)) / np.sum(w * np.abs(y)))


def mase(y, yhat, y_train, m=1):
    """MAE scaled by in-sample MAE of the (seasonal) naive forecast with lag m."""
    y, yhat, y_train = _arr(y, yhat, y_train)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y - yhat)) / scale)


def bias(y, yhat):
    """Mean error (forecast - actual). Positive = we over-forecast."""
    y, yhat = _arr(y, yhat)
    return float(np.mean(yhat - y))


# ---------- interval / quantile metrics ----------
def pinball(y, q_pred, tau):
    """Pinball (quantile) loss for quantile level tau in (0, 1)."""
    y, q_pred = _arr(y, q_pred)
    diff = y - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage(y, lo, hi):
    """Share of actual values inside [lo, hi]."""
    y, lo, hi = _arr(y, lo, hi)
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------- classification metrics ----------
def confusion(y_true, y_score, threshold=0.5):
    t = np.asarray(y_true).astype(int)
    p = (np.asarray(y_score, float) >= threshold).astype(int)
    tp = int(np.sum((p == 1) & (t == 1)))
    fp = int(np.sum((p == 1) & (t == 0)))
    fn = int(np.sum((p == 0) & (t == 1)))
    tn = int(np.sum((p == 0) & (t == 0)))
    return tp, fp, fn, tn


def precision_recall(y_true, y_score, threshold=0.5):
    tp, fp, fn, _ = confusion(y_true, y_score, threshold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec


def roc_auc(y_true, y_score):
    """Probability that a random positive scores higher than a random negative."""
    t = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, float)
    pos, neg = s[t == 1], s[t == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def pr_auc(y_true, y_score):
    """Average precision: mean of precision at the rank of every positive."""
    t = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(y_score, float), kind="stable")
    t = t[order]
    hits = np.cumsum(t)
    precision_at_k = hits / np.arange(1, len(t) + 1)
    return float(np.sum(precision_at_k * t) / t.sum())


def brier(y_true, y_prob):
    t, p = _arr(y_true, y_prob)
    return float(np.mean((p - t) ** 2))


def calibration_table(y_true, y_prob, bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    """Rows of (bin_lo, bin_hi, n, mean_predicted, observed_rate)."""
    t, p = _arr(y_true, y_prob)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append((lo, min(hi, 1.0), int(m.sum()), float(p[m].mean()), float(t[m].mean())))
    return rows


def ece(y_true, y_prob, bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    """Expected calibration error: weighted mean |predicted - observed| over bins."""
    rows = calibration_table(y_true, y_prob, bins)
    n = sum(r[2] for r in rows)
    return float(sum(r[2] / n * abs(r[3] - r[4]) for r in rows))


if __name__ == "__main__":
    # The six hand-worked predictions from section "a worked example" (example data)
    y = [1.00, 1.20, 0.80, 2.50, 0.50, 1.50]      # actual unit price, EUR
    yhat = [1.10, 1.00, 0.90, 2.00, 0.60, 1.60]   # model forecast, EUR
    print(f"MAE   = {mae(y, yhat):.4f}")
    print(f"RMSE  = {rmse(y, yhat):.4f}")
    print(f"MAPE  = {mape(y, yhat):.4f}")
    print(f"sMAPE = {smape(y, yhat):.4f}")
    print(f"WAPE  = {wape(y, yhat):.4f}")
    print(f"Bias  = {bias(y, yhat):+.4f}")

    # MASE needs ONE series: weekly price history of one part (example data)
    hist = [1.00, 1.10, 0.90, 1.00, 1.20, 1.10]
    test, fc = [1.15, 1.05], [1.10, 1.10]
    print(f"MASE  = {mase(test, fc, hist):.4f}")

    # Eight quotes with win-probability scores (example data)
    won = [1, 1, 0, 1, 0, 1, 0, 0]
    p = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    print("confusion (tp, fp, fn, tn) =", confusion(won, p))
    print("precision, recall =", precision_recall(won, p))
    print(f"ROC-AUC = {roc_auc(won, p):.4f}  PR-AUC (AP) = {pr_auc(won, p):.4f}")
    print(f"Brier = {brier(won, p):.4f}")
