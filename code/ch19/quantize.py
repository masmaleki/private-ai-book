"""quantize.py - int8 / int4 quantization by hand with NumPy (teaching only).
Shows scale, zero-point, round-trip error, and why group-wise scales help.
"""
import numpy as np

def quant_symmetric(w, bits=8):
    """Symmetric quantization: zero maps to 0, one scale for the whole tensor."""
    qmax = 2 ** (bits - 1) - 1              # 127 for int8, 7 for int4
    scale = np.abs(w).max() / qmax
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int32)
    return q, scale

def quant_asymmetric(w, bits=8):
    """Asymmetric (affine) quantization with a zero-point."""
    qmin, qmax = 0, 2 ** bits - 1           # 0..255 for uint8
    scale = (w.max() - w.min()) / (qmax - qmin)
    zero_point = int(np.round(qmin - w.min() / scale))
    q = np.clip(np.round(w / scale) + zero_point, qmin, qmax).astype(np.int32)
    return q, scale, zero_point

def quant_groupwise(w, bits=4, group=32):
    """Symmetric quantization with one scale per group of `group` weights."""
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group)
    scales = np.abs(g).max(axis=1, keepdims=True) / qmax
    q = np.clip(np.round(g / scales), -qmax, qmax)
    return (q * scales).reshape(w.shape)    # return dequantized directly

if __name__ == "__main__":
    # 1) The 8 weights from the worked example in the chapter
    w8 = np.array([0.42, -1.30, 0.05, 0.88, -0.31, 1.10, -0.66, 0.19])
    q, s = quant_symmetric(w8, bits=8)
    print("scale        :", round(s, 6))
    print("w / scale    :", np.round(w8 / s, 2).tolist())
    print("int8 values  :", q.tolist())
    deq = q * s
    print("dequantized  :", np.round(deq, 5).tolist())
    print("max abs error:", round(float(np.abs(w8 - deq).max()), 5))

    qa, sa, zp = quant_asymmetric(w8, bits=8)
    print("\nasym scale, zero_point:", round(sa, 6), zp)
    print("uint8 values :", qa.tolist())

    # 2) Error vs bit-width on a synthetic weight matrix with a few outliers
    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.02, size=4096 * 64)          # synthetic weights
    W[rng.choice(W.size, 20, replace=False)] *= 40   # rare large outliers
    print("\nbits  method            mean_abs_err   rel_err")
    for bits in (8, 4):
        q, s = quant_symmetric(W, bits)
        err = np.abs(W - q * s).mean()
        print(f"{bits:>4}  per-tensor        {err:.6f}     {err/np.abs(W).mean():.3f}")
    for group in (128, 32):
        err = np.abs(W - quant_groupwise(W, 4, group)).mean()
        print(f"   4  group={group:<4}        {err:.6f}     "
              f"{err/np.abs(W).mean():.3f}")
