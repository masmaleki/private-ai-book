"""bundle.py - build and verify an OFFLINE release bundle (teaching).

build : python bundle.py build  <src_dir> <bundle_dir>
verify: python bundle.py verify <bundle_dir> <public_key.pem>
The manifest lists every file with size + SHA-256; the manifest itself is
signed with openssl (Ed25519). On the air-gapped side, nothing is installed
unless the signature AND every hash match.
"""
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(src: Path, out: Path, key: Path):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out / "payload")
    files = sorted(p for p in (out / "payload").rglob("*") if p.is_file())
    manifest = {
        "bundle": "private-ai-release",
        "version": "2026.11.0",            # example version
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "target": "tenant-d-airgap",       # fictional tenant
        "files": [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                   "sha256": sha256(p)} for p in files],
    }
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(key),
                    "-rawin", "-in", str(mpath),
                    "-out", str(out / "manifest.sig")], check=True)
    print(f"built {out} with {len(files)} files, manifest signed")


def verify(out: Path, pub: Path) -> bool:
    mpath = out / "manifest.json"
    r = subprocess.run(["openssl", "pkeyutl", "-verify", "-pubin",
                        "-inkey", str(pub), "-rawin", "-in", str(mpath),
                        "-sigfile", str(out / "manifest.sig")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("SIGNATURE INVALID -> reject bundle")
        return False
    print("signature OK")
    ok = True
    for f in json.loads(mpath.read_text())["files"]:
        p = out / f["path"]
        if not p.exists() or sha256(p) != f["sha256"]:
            print("HASH MISMATCH:", f["path"])
            ok = False
    listed = {f["path"] for f in json.loads(mpath.read_text())["files"]}
    extra = [str(p.relative_to(out)) for p in (out / "payload").rglob("*")
             if p.is_file() and str(p.relative_to(out)) not in listed]
    for e in extra:
        print("UNLISTED FILE:", e)
        ok = False
    print("VERIFY RESULT:", "ACCEPT" if ok else "REJECT")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(Path(sys.argv[2]), Path(sys.argv[3]), Path("release_key.pem"))
    elif cmd == "verify":
        sys.exit(0 if verify(Path(sys.argv[2]), Path(sys.argv[3])) else 1)
