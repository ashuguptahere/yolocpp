#!/usr/bin/env bash
# Dependency audit — fails if `CMakeLists.txt` introduces a `find_package`
# or third_party/ entry that isn't on the closed-set whitelist documented
# in `third_party/DEPS.md`. Run as part of #48.

set -euo pipefail
cd "$(dirname "$0")/.."

ALLOWED_FIND_PACKAGE=(Torch CUDAToolkit TensorRT OpenCV)
ALLOWED_TP_DIRS=(CLI11 clay httplib libtorch nccl opencv opencv_root rapidyaml tensorrt tensorrt_root)

violations=0

# 1. find_package() calls must reference a known package.
while IFS= read -r line; do
  pkg=$(echo "$line" | sed -nE 's/.*find_package\(([A-Za-z_][A-Za-z0-9_]*).*/\1/p')
  [[ -z "$pkg" ]] && continue
  ok=0
  for a in "${ALLOWED_FIND_PACKAGE[@]}"; do
    [[ "$pkg" == "$a" ]] && ok=1 && break
  done
  if [[ $ok -eq 0 ]]; then
    echo "[audit] find_package($pkg) at $line — not on closed set"
    violations=$((violations + 1))
  fi
done < <(grep -n "find_package(" CMakeLists.txt cmake/*.cmake 2>/dev/null)

# 2. third_party/ should contain only the documented directories.
for d in third_party/*/; do
  base=$(basename "$d")
  ok=0
  for a in "${ALLOWED_TP_DIRS[@]}"; do
    [[ "$base" == "$a" ]] && ok=1 && break
  done
  if [[ $ok -eq 0 ]]; then
    echo "[audit] third_party/$base/ — undocumented; add to DEPS.md"
    violations=$((violations + 1))
  fi
done

if [[ $violations -gt 0 ]]; then
  echo
  echo "Found $violations audit violation(s). See third_party/DEPS.md."
  exit 1
fi
echo "[ok] dependency audit clean — ${#ALLOWED_FIND_PACKAGE[@]} packages, $(ls third_party/ | wc -l) third_party/ entries"
