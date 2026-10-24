#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
# shellcheck disable=SC1091
source config/pins.env
set +a

CACHE="${NEURO_CACHE:-$ROOT/.cache/neurounner}"
DOWNLOADS="$CACHE/downloads"
RUNTIME_ROOT="$CACHE/runtime"
MODELS="$CACHE/models"
mkdir -p "$DOWNLOADS" "$RUNTIME_ROOT" "$MODELS"

want="${1:-all}"

verify() {
  local file="$1"
  local sha="$2"
  [[ -f "$file" ]] && echo "$sha  $file" | sha256sum -c - >/dev/null 2>&1
}

fetch_verified() {
  local url="$1"
  local dest="$2"
  local sha="$3"
  local label="$4"

  if verify "$dest" "$sha"; then
    echo "[cache hit] $label"
    return 0
  fi

  rm -f "$dest" "$dest.part"
  echo "[download] $label"
  curl -fL --retry 5 --retry-all-errors --connect-timeout 20     --output "$dest.part" "$url"
  echo "$sha  $dest.part" | sha256sum -c -
  mv "$dest.part" "$dest"
}

runtime_dir="$RUNTIME_ROOT/$LLAMA_TAG"
archive="$DOWNLOADS/$LLAMA_ARCHIVE"
marker="$runtime_dir/.archive-sha256"

if [[ ! -f "$marker" ]] || [[ "$(cat "$marker" 2>/dev/null || true)" != "$LLAMA_SHA256" ]]; then
  fetch_verified "$LLAMA_URL" "$archive" "$LLAMA_SHA256" "llama.cpp $LLAMA_TAG Ubuntu x64 CPU"
  rm -rf "$runtime_dir"
  mkdir -p "$runtime_dir"
  tar -xzf "$archive" -C "$runtime_dir"
  printf '%s\n' "$LLAMA_SHA256" > "$marker"
else
  echo "[cache hit] extracted llama.cpp $LLAMA_TAG"
fi

server="$(find "$runtime_dir" -type f -name llama-server -print -quit)"
cli="$(find "$runtime_dir" -type f -name llama-cli -print -quit)"
bench="$(find "$runtime_dir" -type f -name llama-bench -print -quit)"
if [[ -z "$server" || -z "$cli" || -z "$bench" ]]; then
  echo "Pinned llama.cpp archive did not contain required binaries" >&2
  exit 2
fi
chmod +x "$server" "$cli" "$bench"

case "$want" in
  all|sprocket)
    fetch_verified "$SPROCKET_URL" "$MODELS/$SPROCKET_FILE" "$SPROCKET_SHA256" "Sprocket-500M Q4_K_M"
    ;;
esac

case "$want" in
  all|lfm)
    fetch_verified "$LFM_URL" "$MODELS/$LFM_FILE" "$LFM_SHA256" "LFM2.5-1.2B-Instruct QAD-Q4_0"
    ;;
esac

case "$want" in
  all|sprocket|lfm|runtime) ;;
  *)
    echo "usage: $0 [all|runtime|sprocket|lfm]" >&2
    exit 2
    ;;
esac

cat > "$CACHE/paths.env" <<EOF
LLAMA_SERVER=$server
LLAMA_CLI=$cli
LLAMA_BENCH=$bench
SPROCKET_MODEL=$MODELS/$SPROCKET_FILE
LFM_MODEL=$MODELS/$LFM_FILE
EOF

echo "=== pinned runtime ==="
"$cli" --version | head -n 2 || true
echo "llama_commit=$LLAMA_COMMIT"
echo "cache=$CACHE"
du -sh "$CACHE" || true

if [[ "$want" == "all" || "$want" == "sprocket" ]]; then
  echo "$SPROCKET_SHA256  $MODELS/$SPROCKET_FILE" | sha256sum -c -
fi
if [[ "$want" == "all" || "$want" == "lfm" ]]; then
  echo "$LFM_SHA256  $MODELS/$LFM_FILE" | sha256sum -c -
fi
