"""capacity_tco.py - capacity planning + TCO method (teaching).

Every number below is an ASSUMPTION you must replace with your own
measurements, datasheets and quotes. Costs are in abstract "units",
not euros: the point is the method, not the price.
"""
from dataclasses import dataclass

GB = 1e9


@dataclass
class Model:              # an 8B-class decoder with grouped-query attention
    params_b: float = 8.0
    bytes_per_weight: float = 2.0  # bf16; ~0.55 for 4-bit
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    kv_bytes: float = 2.0          # bf16 KV cache

    def weights_gb(self):
        return self.params_b * 1e9 * self.bytes_per_weight / GB

    def kv_bytes_per_token(self):
        # 2 (K and V) x layers x kv_heads x head_dim x bytes
        return 2 * self.layers * self.kv_heads * self.head_dim * self.kv_bytes


@dataclass
class Gpu:
    vram_gb: float = 48.0          # assumption: check your datasheet
    mem_bw_gbs: float = 1000.0     # assumption: memory bandwidth GB/s
    util: float = 0.90             # fraction the server may use
    overhead_gb: float = 3.0       # activations, CUDA context, fragments


def kv_budget_tokens(m: Model, g: Gpu):
    free = g.vram_gb * g.util - m.weights_gb() - g.overhead_gb
    return max(free, 0) * GB / m.kv_bytes_per_token()


def decode_tok_s_per_user(m: Model, g: Gpu, users: int, ctx_tokens: int):
    """Memory-bound estimate: each decode step reads all weights once
    plus the KV cache of every active sequence. Upper bound, not a promise."""
    bytes_per_step = m.weights_gb() * GB + users * ctx_tokens \
        * m.kv_bytes_per_token()
    return g.mem_bw_gbs * GB / bytes_per_step


def peak_concurrency(users, req_per_user_h, out_tokens, tok_s, peak=3.0):
    """Little's law: L = lambda x W, times a peak factor."""
    lam = users * req_per_user_h / 3600          # requests per second
    w = out_tokens / tok_s                        # seconds per request
    return lam * w * peak


def tco(capex, life_months, opex_month, months):
    """On-prem cumulative cash: capex up front + monthly opex
    (power, space, support, people). life_months is returned for
    depreciation views; replace every input with your own quotes."""
    return [capex + opex_month * t for t in range(months + 1)], life_months


def rent(month_price, months):
    return [month_price * t for t in range(months + 1)]


if __name__ == "__main__":
    m, g = Model(), Gpu()
    print(f"weights: {m.weights_gb():.1f} GB")
    print(f"KV per token: {m.kv_bytes_per_token() / 1024:.0f} KiB")
    budget = kv_budget_tokens(m, g)
    print(f"KV budget: {budget:,.0f} tokens "
          f"-> {budget / 4096:.0f} sequences of 4k tokens")
    for u in (1, 4, 8, 16, 32):
        print(f"users={u:>2} ctx=4k -> ~{decode_tok_s_per_user(m, g, u, 4096):5.1f}"
              " tok/s per user (upper bound)")
    c = peak_concurrency(users=25, req_per_user_h=12, out_tokens=600,
                         tok_s=30)
    print(f"peak concurrent requests (25 users): {c:.1f}")
    on_prem, _ = tco(capex=100, life_months=48, opex_month=2.5, months=36)
    cloud = rent(month_price=6.0, months=36)
    be = next((t for t in range(37) if cloud[t] >= on_prem[t]), None)
    print(f"break-even month (units example): {be}")
