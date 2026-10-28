"""vram_calc.py - rough GPU memory budget for serving an LLM (teaching only).

VRAM ~= weights + KV cache + overhead
  weights  = n_params * bytes_per_weight
  KV cache = 2 (K and V) * n_layers * n_kv_heads * head_dim * kv_bytes
             * tokens_per_sequence * concurrent_sequences
  overhead = fixed runtime overhead + activation reserve (assumption: measure it!)
Read n_layers, n_kv_heads (num_key_value_heads), head_dim from the model's config.json.
"""
from dataclasses import dataclass

GB = 1024 ** 3

@dataclass
class Scenario:
    name: str
    n_params_b: float        # billions of parameters
    bits_per_weight: float   # 16 for bf16, ~4.5 for 4-bit incl. scales (assumption)
    n_layers: int
    n_kv_heads: int
    head_dim: int
    ctx_tokens: int          # tokens kept per sequence (prompt + answer)
    concurrent: int          # sequences in flight at the same time
    kv_bytes: float = 2.0    # 2 for fp16/bf16 KV cache, 1 for fp8 KV cache
    fixed_overhead_gb: float = 1.5     # CUDA context, runtime (assumption)
    activation_frac: float = 0.10      # reserve as fraction of weights (assumption)

    def budget(self):
        weights = self.n_params_b * 1e9 * self.bits_per_weight / 8
        kv_per_token = 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_bytes
        kv = kv_per_token * self.ctx_tokens * self.concurrent
        overhead = self.fixed_overhead_gb * GB + self.activation_frac * weights
        return {
            "weights_gb": weights / GB,
            "kv_per_token_kb": kv_per_token / 1024,
            "kv_gb": kv / GB,
            "overhead_gb": overhead / GB,
            "total_gb": (weights + kv + overhead) / GB,
        }

# Example configs (illustrative, "8B-class" / "32B-class"; check the real config.json)
SCENARIOS = [
    Scenario("A: laptop lab, 8B-class, 4-bit, 1 user x 8k",
             8, 4.5, 32, 8, 128, 8192, 1),
    Scenario("B: pilot, 8B-class, bf16, 16 users x 8k",
             8, 16, 32, 8, 128, 8192, 16),
    Scenario("C: sales team, 32B-class, 4-bit, 20 users x 16k",
             32, 4.5, 64, 8, 128, 16384, 20),
    Scenario("C': same as C with fp8 KV cache",
             32, 4.5, 64, 8, 128, 16384, 20, kv_bytes=1.0),
]

if __name__ == "__main__":
    for s in SCENARIOS:
        b = s.budget()
        print(s.name)
        print(f"  W {b['weights_gb']:5.1f} | KV/tok {b['kv_per_token_kb']:4.0f} KB"
              f" | KV {b['kv_gb']:5.1f} | ovh {b['overhead_gb']:4.1f}"
              f" | total {b['total_gb']:5.1f} GB")
