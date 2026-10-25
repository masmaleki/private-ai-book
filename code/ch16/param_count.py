"""param_count.py - count transformer parameters by formula and verify against PyTorch."""
import torch
from minigpt import Config, MiniGPT


def count(d, L, V, mlp_hidden=None, mlp_mats=2, tied=True, kv_ratio=1.0):
    """Decoder-only, no biases, RMSNorm weights only, RoPE (no position params)."""
    h = mlp_hidden or 4 * d
    attn = 2 * d * d + 2 * d * int(d * kv_ratio)      # W_Q, W_O full; W_K, W_V shrink with GQA
    mlp = mlp_mats * d * h
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = V * d
    head = 0 if tied else V * d
    return dict(attn=attn, mlp=mlp, norms=norms, per_layer=per_layer, layers=L * per_layer,
                emb=emb, head=head, final_norm=d, total=L * per_layer + emb + head + d)


if __name__ == "__main__":
    small = count(512, 8, 32_000)
    for k, v in small.items():
        print(f"{k:<11}{v:>15,}")
    print("untied total:", f"{count(512, 8, 32_000, tied=False)['total']:,}")
    # a public 7B-class config (Llama-2-7B style: d=4096, 32 layers, SwiGLU 11008, vocab 32000, untied)
    big = count(4096, 32, 32_000, mlp_hidden=11_008, mlp_mats=3, tied=False)
    print("7B-class total:", f"{big['total']:,}", " per layer:", f"{big['per_layer']:,}")
    # verify the formula against an actual PyTorch module (small dims so it is fast)
    c = Config(vocab_size=32_000, d_model=512, n_layers=8, n_heads=8)
    m = MiniGPT(c)
    print("PyTorch MiniGPT(d=512, L=8, V=32k):", f"{sum(p.numel() for p in m.parameters()):,}")
