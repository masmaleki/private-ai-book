#!/usr/bin/env bash
# egress_check.sh - run INSIDE the LLM host/container. Every probe must FAIL.
# Exit code 0 = no egress observed, 1 = something reached the internet.
set -u
leak=0
for url in https://huggingface.co https://pypi.org https://example.com; do
  if curl -s --max-time 5 -o /dev/null "$url"; then
    echo "LEAK: reached $url"; leak=1
  else
    echo "ok: blocked $url"
  fi
done
if getent hosts example.com >/dev/null 2>&1; then
  echo "WARN: public DNS resolves (check resolver policy)"
fi
echo "established outbound connections (should be internal only):"
ss -tn state established 2>/dev/null \
  | awk 'NR>1 {print "  " $3 " -> " $4}' | head -n 20
exit $leak
