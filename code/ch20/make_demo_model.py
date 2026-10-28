"""make_demo_model.py - create a tiny FAKE model folder for practising checks.
Writes a valid .safetensors file with NumPy only (no torch needed).
"""
import json, struct, hashlib
from pathlib import Path
import numpy as np

def write_safetensors(path, tensors, metadata=None):
    """Minimal safetensors writer: 8-byte LE header length + JSON header + raw bytes."""
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = arr.astype(np.float16).tobytes()
        header[name] = {"dtype": "F16", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw); offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    hjson = json.dumps(header).encode()
    hjson += b" " * (-len(hjson) % 8)          # pad header to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson))); f.write(hjson)
        for b in blobs: f.write(b)

if __name__ == "__main__":
    d = Path("demo-model"); d.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)                     # synthetic weights
    write_safetensors(d / "model.safetensors", {
        "embed.weight": rng.normal(size=(64, 16)),
        "layers.0.attn.q.weight": rng.normal(size=(16, 16)),
        "lm_head.weight": rng.normal(size=(64, 16)),
    }, metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({
        "model_type": "demo", "num_hidden_layers": 1, "hidden_size": 16,
        "torch_dtype": "float16"}, indent=2))
    (d / "tokenizer.json").write_text('{"version": "demo"}')
    # The manifest we would have recorded when the model entered our mirror
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(d.iterdir()) if p.is_file()}
    Path("demo-model.sha256.json").write_text(json.dumps(manifest, indent=2))
    print("created", d, "with", len(manifest), "files")
