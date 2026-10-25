"""Chapter 9: seasonal naive and Croston by hand (fictional example numbers)."""
import numpy as np


def croston(y, alpha=0.2, verbose=True):
    z = p = None
    q = 1                 # periods since last demand
    out = []
    for t, v in enumerate(y):
        f = None if z is None else z / p
        out.append(f)     # forecast made before seeing y[t]
        if v > 0:
            if z is None:
                z, p = float(v), float(t + 1)
            else:
                z += alpha * (v - z)
                p += alpha * (q - p)
            if verbose:
                print(f"  month {t + 1:>2}: demand {v:>3}  interval {q}  "
                      f"z={z:.2f}  p={p:.3f}  forecast={z / p:.2f}")
            q = 1
        else:
            q += 1
    return z / p, out


if __name__ == "__main__":
    # 1) seasonal naive on one smooth part, units per month (example data)
    y2024 = np.array([400, 420, 470, 430, 410, 400, 390, 280, 450, 450, 430, 340])
    y2025 = np.array([420, 440, 500, 450, 420, 430, 400, 300, 470, 480, 440, 350])
    snaive = y2024                                   # same month last year
    naive = np.r_[y2024[-1], y2025[:-1]]             # last month's actual (1 step ahead)
    print("== 1) 2025 forecasts ==")
    print("seasonal naive errors:", np.abs(y2025 - snaive))
    print("naive errors:         ", np.abs(y2025 - naive))
    print(f"MAE seasonal naive = {np.abs(y2025 - snaive).mean():.2f}")
    print(f"MAE naive (1 step) = {np.abs(y2025 - naive).mean():.2f}")

    # 2) Croston on a sparse part (example data)
    sparse = np.array([0, 0, 300, 0, 0, 0, 250, 0, 0, 350, 0, 0])
    print("\n== 2) Croston, alpha = 0.2 ==")
    f, _ = croston(sparse)
    print(f"Croston forecast per month: {f:.2f}")
    print(f"SBA (x (1 - alpha/2) = 0.9): {0.9 * f:.2f}")
    print(f"simple mean of 12 months: {sparse.mean():.2f}")
    ma3 = [sparse[i - 3:i].mean() for i in range(3, 13)]
    print("MA(3) forecasts:", np.round(ma3, 1))
