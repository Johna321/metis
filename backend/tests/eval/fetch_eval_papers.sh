#!/usr/bin/env bash
# Download eval papers from arXiv into backend/tests/eval/papers/
set -euo pipefail
DIR="$(dirname "$0")/papers"
mkdir -p "$DIR"
download() {
    local name=$1 url=$2
    if [ ! -f "$DIR/$name" ]; then
        echo "Fetching $name..."
        curl -L "$url" -o "$DIR/$name"
    else
        echo "$name already present, skipping"
    fi
}
download "late_chunking.pdf" "https://arxiv.org/pdf/2409.04701"
download "attention_is_all_you_need.pdf" "https://arxiv.org/pdf/1706.03762"
download "hichunk.pdf" "https://arxiv.org/pdf/2509.11552"
