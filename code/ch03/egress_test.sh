#!/usr/bin/env bash
# egress_test.sh - Chapter 3: minimal "cut the cable" egress probe.
# Every public target must FAIL on a sealed AI server. Exit 1 on any leak.
set -u
TARGETS=("https://huggingface.co" "https://pypi.org/simple/" "https://example.com"
         "http://1.1.1.1")
leaks=0
for url in "${TARGETS[@]}"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url")
  rc=$?
  if [[ $rc -eq 0 && $code != "000" ]]; then
    echo "LEAK    $url (http $code)"; leaks=$((leaks+1))
  else
    echo "blocked $url (curl rc=$rc)"
  fi
done
# DNS is a channel too: a sealed server should not resolve public names
if getent hosts example.com >/dev/null 2>&1; then
  echo "LEAK    DNS resolves public names"; leaks=$((leaks+1))
else
  echo "blocked DNS lookup of public names"
fi
[[ $leaks -eq 0 ]] && echo "PASS" || { echo "FAIL: $leaks leak(s)"; exit 1; }
