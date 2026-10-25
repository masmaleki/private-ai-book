"""costs.py - chapter 16 arithmetic: training FLOPs (C ~ 6*N*D), GPU-hours under stated
assumptions, Chinchilla-style token budget, perplexity and a toy DPO loss.
All hardware numbers are ASSUMPTIONS for illustration, not vendor quotes."""
import math

PEAK_FLOPS = 1.0e15      # assumption: ~1 PFLOP/s dense bf16 per accelerator (check datasheet)
MFU = 0.40               # assumption: 40 % model FLOPs utilisation
EUR_PER_GPU_HOUR = 2.5   # assumption: illustrative rental price, NOT a quote


def train_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens


def gpu_hours(flops, peak=PEAK_FLOPS, mfu=MFU):
    return flops / (peak * mfu) / 3600


def perplexity(mean_nll):
    return math.exp(mean_nll)


def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta=0.1):
    """lp_* = log-prob of chosen (w) / rejected (l) answer under policy, ref_* under reference."""
    margin = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -math.log(1 / (1 + math.exp(-margin))), margin


if __name__ == "__main__":
    for name, n, d in [("7B/2T", 7e9, 2e12), ("7B/15T", 7e9, 15e12), ("70B/15T", 70e9, 15e12),
                       ("mini-GPT", 448_704, 3000 * 32 * 64)]:
        c = train_flops(n, d)
        h = gpu_hours(c)
        print(f"{name:<9} C={c:.1e} GPU-h={h:>9,.0f} EUR~{h * EUR_PER_GPU_HOUR:>10,.0f}"
              f" days@512={h / 512 / 24:,.1f}")
    print("chinchilla tokens for 7B (20 tok/param):", f"{20 * 7e9:.1e}")
    print("domain continued pretraining 7B on 2B tokens:",
          f"{gpu_hours(train_flops(7e9, 2e9)):,.0f} GPU-hours")
    for nll in (4.043, 0.178, 0.1347, 2.0):
        print(f"mean NLL {nll} -> perplexity {perplexity(nll):.3f}")
    print("ln(57) =", round(math.log(57), 4))
    for case in [(-12.0, -15.0, -13.0, -14.0), (-11.0, -18.0, -13.0, -14.0),
                 (-14.0, -13.0, -13.0, -14.0)]:
        loss, m = dpo_loss(*case)
        print("dpo", case, f"margin={m:.2f} loss={loss:.4f}")
