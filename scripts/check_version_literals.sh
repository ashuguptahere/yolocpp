#!/usr/bin/env bash
# Lint check for stray X.Y.Z version literals outside the allow-list.
#
# Single source of truth for the project version is ./VERSION. CMake
# reads it, embeds it via config.hpp -> YOLOCPP_VERSION_STRING, surfaces
# it through `yolocpp --version` / `yolocpp info`. Anywhere else that
# hardcodes a 0.MINOR.PATCH literal is a maintenance hazard — when the
# version bumps, those copies go stale silently.
#
# Allow-list (these places are *expected* to contain a literal):
#   - VERSION                     — the source of truth
#   - CMakeLists.txt              — historical comment about the file
#   - CHANGELOG.md                — release headings (immutable)
#   - SESSION_DIGEST.md           — frozen-in-time snapshot (per-session)
#   - TODO.md "landed in X.Y.Z"  — historical references (immutable)
#   - README.md table cells "added 0.X.Y" — same, historical
#   - third_party/, build/        — vendored / generated
#
# Anything else that matches `0\.\d+\.\d+` is flagged. Run as part of
# pre-commit / CI; non-zero exit means a stale literal slipped in.
#
# Usage:
#   bash scripts/check_version_literals.sh           # whole repo
#   bash scripts/check_version_literals.sh --strict  # also flag historical
#                                                    # 0.X.Y in TODO/README

set -euo pipefail

cd "$(dirname "$0")/.."

# Pattern: 0.MINOR.PATCH not surrounded by another digit/dot (Perl
# lookarounds rule out IP-style 4-octet sequences `127.0.0.1` whose
# trailing `0.0.1` would otherwise word-boundary match). We're pre-1.0
# so the leading digit is `0`; revisit the regex when 1.0.0 ships.
PATTERN='(?<![0-9.])0\.[0-9]+\.[0-9]+(?![0-9.])'

# Files that are allowed to carry version literals.
ALLOW_FILES=(
  "^VERSION$"
  "^CHANGELOG\.md$"
  "^SESSION_DIGEST\.md$"
  "^scripts/check_version_literals\.sh$"
)

# Directories to skip entirely.
EXCLUDE_DIRS=(
  "^third_party/"
  "^build/"
  "^\.git/"
  "^runs/"
)

# Inline patterns inside otherwise-allowed lines: historical refs +
# third-party / vendor / non-yolocpp version mentions.
ALLOWED_LINE_PATTERNS=(
  # Historical "landed in X.Y.Z" / "added 0.20.0" — references to a
  # specific past commit, immutable.
  "landed in 0\."
  "added in 0\."
  "added 0\."
  "in 0\.[0-9]+\.[0-9]+ "
  "in 0\.[0-9]+\.[0-9]+\)"
  "from 0\."
  "since 0\."
  "pre-0\."
  "post-0\."
  # Upstream / third-party version references — not yolocpp.
  "[Mm]eituan"
  "rapidyaml"
  "producer_version"
  "[Yy][Oo][Ll][Oo][Vv]6"
  "ultralytics/assets"
  "v0\.0\.0"
  "[Rr]elease 0\."
  # Default IPs / network identifiers that happen to embed 0.0.1.
  "127\.0\.0\.1"
  "MASTER_ADDR"
  # Acceptable historical bare parenthetical "(0.X.Y)" only if line
  # also mentions a verb like "training", "fix", etc. — too loose;
  # require explicit qualifier words instead. (No entry here.)
)

violations=0
files=$(git ls-files | grep -E '\.(md|txt|cpp|hpp|cc|cmake|in|sh|py|yaml|yml)$' || true)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue

  # Skip excluded dirs.
  skip=0
  for d in "${EXCLUDE_DIRS[@]}"; do
    if [[ "$f" =~ $d ]]; then skip=1; break; fi
  done
  [[ $skip -eq 1 ]] && continue

  # Skip allow-list files entirely.
  for af in "${ALLOW_FILES[@]}"; do
    if [[ "$f" =~ $af ]]; then skip=1; break; fi
  done
  [[ $skip -eq 1 ]] && continue

  # Find candidate lines, then drop those matching an allowed pattern.
  while IFS= read -r line; do
    lineno="${line%%:*}"
    rest="${line#*:}"

    keep=1
    for ap in "${ALLOWED_LINE_PATTERNS[@]}"; do
      if echo "$rest" | grep -qE "$ap"; then keep=0; break; fi
    done

    if [[ $keep -eq 1 ]]; then
      echo "[stale-version] $f:$lineno: $rest"
      violations=$((violations + 1))
    fi
  done < <(grep -nP "$PATTERN" "$f" 2>/dev/null || true)
done <<< "$files"

if [[ $violations -gt 0 ]]; then
  echo
  echo "Found $violations stray version literal(s) outside the allow-list."
  echo "Single source of truth: ./VERSION (read by CMake -> config.hpp -> YOLOCPP_VERSION_STRING)."
  echo "Allowed historical mentions must use one of: 'landed in X.Y.Z', 'added in X.Y.Z', 'added X.Y.Z', 'in X.Y.Z ', 'pre-X.Y', etc."
  exit 1
fi

echo "[ok] no stray version literals found"
