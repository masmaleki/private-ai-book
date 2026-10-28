"""verify_model.py - supply-chain checks for a downloaded model folder.

1. SHA-256 of every file vs. a trusted manifest (recorded at import time)
2. refuse pickle-based weight files (.bin, .pt, .pth, .pkl, .ckpt)
3. read safetensors headers without loading weights (dtype, shapes, #params)
4. flag configs that require remote code (auto_map / trust_remote_code)
Usage: python verify_model.py <model_dir> <manifest.json>
"""
import hashlib, json, struct, sys
from pathlib import Path

PICKLE_EXT = {".bin", ".pt", ".pth", ".pkl", ".ckpt"}

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):      # stream: model files can be huge
            h.update(block)
    return h.hexdigest()

def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 * 1024 * 1024:
            raise ValueError("header too large, refusing")
        return json.loads(f.read(n))

def main(model_dir, manifest_path):
    model_dir = Path(model_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    problems = []
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    # 1) hashes
    for p in files:
        rel = str(p.relative_to(model_dir))
        if rel not in manifest:
            problems.append(f"UNEXPECTED FILE: {rel}")
        elif sha256(p) != manifest[rel]:
            problems.append(f"HASH MISMATCH: {rel}")
        else:
            print(f"ok   sha256  {rel}")
    for rel in manifest:
        if not (model_dir / rel).exists():
            problems.append(f"MISSING FILE: {rel}")
    # 2) pickle files
    for p in files:
        if p.suffix in PICKLE_EXT:
            problems.append(f"PICKLE WEIGHTS (can run code on load): {p.name}")
    # 3) safetensors metadata
    total = 0
    for p in files:
        if p.suffix == ".safetensors":
            hdr = safetensors_header(p)
            for name, info in hdr.items():
                if name == "__metadata__":
                    continue
                n = 1
                for d in info["shape"]:
                    n *= d
                total += n
                print(f"     tensor  {name:<26} {info['dtype']:<5} {info['shape']}")
    print(f"     params  {total:,}")
    # 4) remote code
    cfg = model_dir / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        if "auto_map" in c:
            problems.append("config.json has auto_map -> needs trust_remote_code")
    print("\nRESULT:", "PASS" if not problems else "FAIL")
    for pr in problems:
        print("  -", pr)
    return 0 if not problems else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
