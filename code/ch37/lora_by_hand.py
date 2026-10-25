"""lora_by_hand.py - the low-rank idea with plain NumPy (teaching only).

1) A 4x4 "weight update" that is really rank 1 = (4x1) @ (1x4).
2) Parameter counts: full update vs LoRA for a 4096x4096 layer.
3) A LoRA forward pass: y = W x + (alpha / r) * B (A x).
"""
import numpy as np

# 1) A rank-1 update: 16 numbers described by only 4 + 4 = 8 numbers
b = np.array([[1.0], [2.0], [0.0], [-1.0]])  # 4x1
a = np.array([[0.5, 0.0, 1.0, 2.0]])          # 1x4
delta_w = b @ a                               # 4x4
print("delta_W =\n", delta_w)
print("rank(delta_W) =", np.linalg.matrix_rank(delta_w))


# 2) Parameter counting
def lora_params(d_out, d_in, r):
    """A is r x d_in, B is d_out x r."""
    return r * d_in + d_out * r


full = 4096 * 4096
for r in (4, 8, 16, 64):
    p = lora_params(4096, 4096, r)
    print(f"r={r:>3}: LoRA params={p:>9,}  full={full:,}  "
          f"ratio={p / full:.4%}")

# 3) LoRA forward pass on a tiny layer
rng = np.random.default_rng(42)
d, r, alpha = 6, 2, 4
W = rng.normal(size=(d, d))            # frozen pretrained weight
A = rng.normal(scale=0.1, size=(r, d))  # trainable, random init
B = np.zeros((d, r))                    # trainable, ZERO init
x = rng.normal(size=(d,))

y_base = W @ x
y_lora = W @ x + (alpha / r) * (B @ (A @ x))
print("at init, LoRA output == base output:", np.allclose(y_base, y_lora))

# pretend training moved B a bit
B = rng.normal(scale=0.1, size=(d, r))
y_lora = W @ x + (alpha / r) * (B @ (A @ x))
print("after 'training', max change:", np.abs(y_lora - y_base).max().round(4))

# merging: W' = W + (alpha/r) * B @ A gives the same output, no extra cost
W_merged = W + (alpha / r) * (B @ A)
print("merged output identical:", np.allclose(W_merged @ x, y_lora))
