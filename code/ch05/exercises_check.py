"""Check the numeric answers of the chapter 5 exercises (example data only)."""
import numpy as np

# 5-1: dot product and cosine of two (example) capacitor spec vectors
p = np.array([3.3, 0.5, 1.0]); q = np.array([5.0, 0.5, 1.0]); r = np.array([0.3, 2.0, 0.0])
cos = lambda x, y: x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
print("5-1 p.q =", round(p @ q, 4), "p.r =", round(p @ r, 4))
print("    |p| =", round(np.linalg.norm(p), 4), "|q| =", round(np.linalg.norm(q), 4),
      "|r| =", round(np.linalg.norm(r), 4))
print("    cos(p,q) =", round(cos(p, q), 4), "cos(p,r) =", round(cos(p, r), 4))

# 5-2: L1 and L2 distances
print("5-2 L2(p-q) =", round(np.linalg.norm(p - q), 4), "L1(p-q) =", np.abs(p - q).sum())
print("    L2(p-r) =", round(np.linalg.norm(p - r), 4), "L1(p-r) =", round(np.abs(p - r).sum(), 4))

# 5-3: matrix as transformation
W = np.array([[1.0, 0.5], [0.0, 2.0]]); v = np.array([4.0, 2.0])
print("5-3 W@v =", W @ v)

# 5-4: mean, median, std with an outlier
x = np.array([10, 12, 11, 13, 54])
print("5-4 mean =", x.mean(), "median =", np.median(x), "var =", x.var(),
      "std =", round(x.std(), 2))
y = x[:4]
print("    without 54: mean =", y.mean(), "std =", round(y.std(), 3))

# 5-5: expected value of two supplier choices
ev_a = 0.9 * 0 + 0.1 * 4000
ev_b = 0.98 * 0 + 0.02 * 4000
print("5-5 expected delay cost A =", ev_a, "B =", ev_b, "net A =", 1000 + ev_a,
      "net B =", 1250 + ev_b)

# 5-6: normal quantiles
mu, s = 2.40, 0.15
print("5-6 P10 =", round(mu - 1.2816 * s, 3), "P90 =", round(mu + 1.2816 * s, 3),
      "95% =", round(mu - 1.96 * s, 3), round(mu + 1.96 * s, 3))

# 5-7: logs
print("5-7 ln(1.25) =", round(np.log(1.25), 4), "exp(0.2)-1 =", round(np.exp(0.2) - 1, 4),
      "ln(0.8) =", round(np.log(0.8), 4))

# 5-8: gradient descent on (w-5)^2
for lr in (0.25, 1.1):
    w = 1.0
    out = []
    for _ in range(3):
        g = 2 * (w - 5); w = w - lr * g; out.append((g, round(w, 4), round((w - 5)**2, 4)))
    print(f"5-8 lr={lr}:", out)

# 5-9: softmax with temperature
def softmax(z, T=1.0):
    e = np.exp(np.array(z) / T); return e / e.sum()
print("5-9 softmax([1,2,3]) =", softmax([1, 2, 3]).round(4))
print("    T=2:", softmax([1, 2, 3], 2).round(4), " T=0.5:", softmax([1, 2, 3], 0.5).round(4))
print("    softmax([101,102,103]) =", softmax([101, 102, 103]).round(4))

# 5-10: cross-entropy
P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]); t = [0, 1, 0]
ce = -np.log(P[np.arange(3), t])
print("5-10 CE per row =", ce.round(4), "mean =", round(ce.mean(), 4))
