"""Answer code for exercises 14-5 and 14-6 (random toy data)."""
import numpy as np
import torch
import torch.nn.functional as F
from attention import softmax, attention, multi_head


# ---- 14-5: padding mask ----
def attention_masked(Q, K, V, causal=True, key_padding=None):
    """Q, K, V: (b, n, d). key_padding: (b, n) bool, True = padding token."""
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)          # (b, n, n)
    n = scores.shape[-1]
    block = np.zeros_like(scores, dtype=bool)
    if causal:
        block |= np.triu(np.ones((n, n), dtype=bool), 1)[None]
    if key_padding is not None:
        block |= key_padding[:, None, :]                          # nobody looks at pads
    scores = np.where(block, -np.inf, scores)
    return softmax(scores) @ V


rng = np.random.default_rng(1)
Q, K, V = (rng.normal(size=(2, 4, 3)) for _ in range(3))
pad = np.array([[False, False, False, False],
                [False, False, True, True]])                      # 2nd sentence: 2 real tokens
mine = attention_masked(Q, K, V, key_padding=pad)
allowed = ~(torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)[None] | torch.tensor(pad)[:, None, :])
ref = F.scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V),
                                     attn_mask=allowed)
print("max diff on real tokens:", np.abs(ref.numpy() - mine)[~pad].max())


# ---- 14-6: grouped-query attention ----
def gqa(X, Wq, Wk, Wv, Wo, n_heads, n_kv_heads, causal=True):
    n, d = X.shape
    dh = d // n_heads
    group = n_heads // n_kv_heads                 # query heads per shared K/V head
    Q, K, V = X @ Wq, X @ Wk, X @ Wv              # K, V: (n, n_kv_heads * dh)
    outs = []
    for h in range(n_heads):
        g = h // group
        q = Q[:, h * dh:(h + 1) * dh]
        k = K[:, g * dh:(g + 1) * dh]
        v = V[:, g * dh:(g + 1) * dh]
        outs.append(attention(q, k, v, causal)[0])
    return np.concatenate(outs, -1) @ Wo


rng = np.random.default_rng(0)
n, d, H = 5, 8, 4
X = rng.normal(size=(n, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
print("GQA(kv=H) == MHA:", np.allclose(gqa(X, Wq, Wk, Wv, Wo, H, H), multi_head(X, Wq, Wk, Wv, Wo, H)))
Wk2, Wv2 = Wk[:, :2 * (d // H)], Wv[:, :2 * (d // H)]       # only 2 K/V heads
print("GQA output shape with 2 kv heads:", gqa(X, Wq, Wk2, Wv2, Wo, H, 2).shape)
