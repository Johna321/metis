#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCHEMA_DIR="$ROOT/crates/metis-typegen/schemas"

# Output paths — change these to relocate generated files
TS_OUTPUT="$ROOT/metis-frontend/src/backend/generated.ts"
PY_OUTPUT="$ROOT/backend/src/metis/core/generated_types.py"

echo "==> Generating JSON Schemas from Rust types..."
cargo run -p metis-typegen

echo "==> Generating TypeScript types..."
cd "$ROOT/metis-frontend"
{
  cat <<'HEADER'
/* eslint-disable */
/**
 * AUTO-GENERATED — do not edit by hand.
 * Source: crates/metis-types/src/lib.rs
 * Run ./scripts/sync-types.sh to regenerate.
 */

HEADER
  # Process schemas in dependency order (BboxSelection before ChatRequest)
  for schema in BboxSelection IngestResponse VectorizeResponse EvidenceItem ChatRequest ChatStreamEvent; do
    bun run json2ts \
      -i "$SCHEMA_DIR/${schema}.json" \
      --no-additionalProperties \
      --no-declareExternallyReferenced \
      2>/dev/null \
    | sed '/\/\*/,/\*\//d' \
    | sed '/^$/N;/^\n$/d'
    echo ''
  done
} > "$TS_OUTPUT"
cd "$ROOT"

echo "==> Generating Python (Pydantic v2) types..."
uv run --project "$ROOT/backend" datamodel-codegen \
  --input "$SCHEMA_DIR/_combined.json" \
  --output "$PY_OUTPUT" \
  --output-model-type pydantic_v2.BaseModel \
  --target-python-version 3.12

echo "==> Done! Generated files:"
echo "    - $TS_OUTPUT"
echo "    - $PY_OUTPUT"
