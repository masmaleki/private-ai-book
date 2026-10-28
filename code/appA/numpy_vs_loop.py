"""Appendix A: Python loop vs NumPy vectorised code (timings vary per machine)."""
import time

import numpy as np

rng = np.random.default_rng(42)
results = []
for n in (10_000, 100_000, 1_000_000):
    price = rng.uniform(0.1, 5.0, n)          # example unit prices in EUR
    qty = rng.integers(100, 50_000, n)        # example quantities

    t0 = time.perf_counter()
    total_loop = 0.0
    for p, q in zip(price, qty):              # plain Python loop
        total_loop += p * q
    t_loop = time.perf_counter() - t0

    t_vec = float("inf")
    for _ in range(5):                        # best of 5: NumPy calls are very short
        t0 = time.perf_counter()
        total_vec = float((price * qty).sum())   # vectorised: no Python loop
        t_vec = min(t_vec, time.perf_counter() - t0)

    same = np.isclose(total_loop, total_vec)
    results.append((n, t_loop, t_vec))
    print(f"n={n:>9,}  loop={t_loop * 1000:6.1f} ms  numpy={t_vec * 1000:5.2f} ms  "
          f"x{t_loop / t_vec:.0f}  same={same}")

if __name__ == "__main__":
    from fa_plot import fa, plt, save
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ns = [r[0] for r in results]
    ax.plot(ns, [r[1] * 1000 for r in results], "o-", color="#c0504d",
            label=fa("حلقه‌ی پایتون"))
    ax.plot(ns, [r[2] * 1000 for r in results], "s-", color="#1f4e79",
            label=fa("NumPy", False))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(fa("تعداد ردیف")); ax.set_ylabel(fa("زمان (میلی‌ثانیه، مقیاس لگاریتمی)"))
    ax.legend(); ax.set_title(fa("جمع ارزش Offerها: حلقه در برابر برداری (روی سیستم ما)"))
    save(fig, "appA", "numpy_vs_loop.png")
