#!/usr/bin/env bash
# demo.sh - end-to-end demo of the offline bundle (all files are fake).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf demo_src demo_bundle release_key.pem release_pub.pem
# 1) signing key (in reality: kept offline / in an HSM, never in the repo)
openssl genpkey -algorithm ed25519 -out release_key.pem
openssl pkey -in release_key.pem -pubout -out release_pub.pem
# 2) fake payload: model adapter, container image tarball, SBOM
mkdir -p demo_src/models demo_src/images demo_src/sbom
head -c 200000 /dev/urandom > demo_src/models/adapter_model.safetensors
head -c 300000 /dev/urandom > demo_src/images/llm-server_2026.11.tar
echo '{"bomFormat":"CycloneDX","specVersion":"1.5","components":[]}' \
  > demo_src/sbom/llm-server.cdx.json
# 3) build + verify (good case)
python3 bundle.py build demo_src demo_bundle
python3 bundle.py verify demo_bundle release_pub.pem
# 4) tamper: flip one byte in the image tarball, verify again
printf 'X' | dd of=demo_bundle/payload/images/llm-server_2026.11.tar \
  bs=1 seek=100 conv=notrunc status=none
python3 bundle.py verify demo_bundle release_pub.pem || true
