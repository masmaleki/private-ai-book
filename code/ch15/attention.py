"""attention.py - scaled dot-product attention from scratch in NumPy, plus a PyTorch check.
Toy numbers only (example data)."""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)      # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, causal=False):
    """Q: (n, d_k), K: (n, d_k), V: (n, d_v). Returns output (n, d_v) and weights (n, n)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if causal:
        n = scores.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)   # True above the diagonal = future
        scores = np.where(mask, -np.inf, scores)
    w = softmax(scores)
    return w @ V, w, scores


def multi_head(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    """X: (n, d_model). All W: (d_model, d_model). Splits d_model into n_heads heads."""
    n, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    outs = []
    for h in range(n_heads):
        s = slice(h * dh, (h + 1) * dh)
        o, _, _ = attention(Q[:, s], K[:, s], V[:, s], causal)
        outs.append(o)
    return np.concatenate(outs, axis=-1) @ Wo


# ---- the hand example: "need 500 STM32", d = 2 ----
tokens = ["need", "500", "STM32"]
X = np.array([[1., 0.], [0., 1.], [1., 1.]])
Wq = np.array([[1., 0.], [1., 1.]])
Wk = np.eye(2)
Wv = np.array([[1., 0.], [0., 2.]])
Q, K, V = X @ Wq, X @ Wk, X @ Wv

if __name__ == "__main__":
    print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)
    for causal in (False, True):
        out, w, sc = attention(Q, K, V, causal)
        print(f"\n=== causal={causal} ===")
        print("scaled scores =\n", sc)
        print("weights =\n", w)
        print("output =\n", out)

    # ---- multi-head on random data ----
    rng = np.random.default_rng(0)
    n, d, H = 5, 8, 2
    Xr = rng.normal(size=(n, d))
    Ws = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]
    Y = multi_head(Xr, *Ws, n_heads=H)
    print("\nmulti-head output shape:", Y.shape)

    # ---- PyTorch check ----
    import torch
    import torch.nn.functional as F
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    ref = F.scaled_dot_product_attention(t(Q)[None], t(K)[None], t(V)[None], is_causal=True)[0]
    mine, _, _ = attention(Q, K, V, causal=True)
    print("PyTorch SDPA (causal) =\n", ref.numpy())
    print("max |numpy - torch| =", float(np.abs(ref.numpy() - mine).max()))

    mha = torch.nn.MultiheadAttention(d, H, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([t(Ws[0]).T, t(Ws[1]).T, t(Ws[2]).T]))
        mha.out_proj.weight.copy_(t(Ws[3]).T)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        ref2, _ = mha(t(Xr)[None], t(Xr)[None], t(Xr)[None], attn_mask=mask)
    print("max |numpy multi-head - torch MHA| =", float(np.abs(ref2[0].numpy() - Y).max()))
