"""kvcache.py - add a KV cache to the chapter-15 mini-GPT (same weights) and compare
per-token decode time with naive re-computation. CPU only, synthetic model.
Usage: python3 kvcache.py   (needs ../ch16/minigpt_long.pt)"""
import sys, time
sys.path.insert(0, "../ch15")
import torch
import torch.nn.functional as F
from minigpt import load

torch.set_num_threads(4)
m, tok = load("../ch16/minigpt_long.pt")


def rope_at(x, pos):
    """RoPE for x: (B, H, T, Dh) whose tokens sit at absolute positions pos (T,)."""
    half = x.shape[-1] // 2
    freqs = 10000 ** (-torch.arange(half, dtype=torch.float32) / half)
    ang = pos.float()[:, None] * freqs[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


@torch.no_grad()
def step_with_cache(ids, cache):
    """Run only the NEW tokens `ids` (1, T_new); cache holds K, V of earlier tokens."""
    start = cache[0][0].shape[2] if cache[0] else 0
    pos = torch.arange(start, start + ids.shape[1])
    x = m.tok(ids)
    for li, b in enumerate(m.blocks):
        h = b.n1(x)
        B, T, D = h.shape
        q, k, v = b.attn.qkv(h).split(D, dim=-1)
        q, k, v = (t.view(B, T, b.attn.h, D // b.attn.h).transpose(1, 2) for t in (q, k, v))
        q, k = rope_at(q, pos), rope_at(k, pos)
        if cache[li]:                                   # append to the stored keys/values
            k = torch.cat([cache[li][0], k], 2)
            v = torch.cat([cache[li][1], v], 2)
        cache[li] = (k, v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(start == 0 and T > 1))
        x = x + b.attn.out(y.transpose(1, 2).reshape(B, T, D))
        x = x + b.mlp(b.n2(x))
    return m.head(m.norm(x))


if __name__ == "__main__":
    prompt = "please quote 500 pcs STM32F407VGT6, lead time 8 weeks. we need "
    ids = torch.tensor([tok.encode(prompt)])
    # 1) correctness: cached step-by-step logits == full forward logits
    cache = [None] * len(m.blocks)
    lg = [step_with_cache(ids[:, :1], cache)]
    for t in range(1, ids.shape[1]):
        lg.append(step_with_cache(ids[:, t:t + 1], cache))
    full = m(ids)[0]
    print("max |cached - full| logits:", f"{(torch.cat(lg, 1) - full).abs().max():.2e}")
    # 2) timing per generated token at different context lengths
    for ctx in (64, 128, 256, 512):
        x = torch.randint(0, len(tok.chars), (1, ctx))
        t0 = time.perf_counter()
        for _ in range(30):
            m(x)                                        # naive: recompute all ctx tokens
        naive = (time.perf_counter() - t0) / 30
        cache = [None] * len(m.blocks)
        step_with_cache(x, cache)                       # prefill fills the cache
        nxt = x[:, -1:]
        t0 = time.perf_counter()
        for _ in range(30):
            step_with_cache(nxt, cache)                 # decode: only one new token
        cached = (time.perf_counter() - t0) / 30
        print(f"ctx {ctx:4d}: naive {naive * 1e3:6.2f} ms/token  cached {cached * 1e3:5.2f} ms/token")
