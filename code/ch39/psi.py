"""psi.py - Population Stability Index (PSI) for data drift (teaching)."""
import numpy as np


def psi(expected, actual, eps=1e-6):
    """PSI = sum((a - e) * ln(a / e)) over bins; inputs are bin shares."""
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_from_samples(ref, cur, bins=10):
    """Bin edges come from the REFERENCE quantiles, then compare shares."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(cur, edges)[0] / len(cur)
    return psi(e, a)


if __name__ == "__main__":
    # worked example from the chapter: 4 quantity buckets (example data)
    ref = [0.40, 0.30, 0.20, 0.10]
    cur = [0.25, 0.30, 0.25, 0.20]
    print("PSI by hand example:", round(psi(ref, cur), 4))

    rng = np.random.default_rng(42)  # synthetic log-quantities
    train = rng.normal(6.0, 1.0, 5000)
    for shift in (0.0, 0.2, 0.5, 1.0):
        live = rng.normal(6.0 + shift, 1.0, 2000)
        print(f"mean shift {shift:.1f} -> PSI {psi_from_samples(train, live):.3f}")
