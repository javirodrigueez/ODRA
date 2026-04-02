#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  evaluation/run_all_metrics.sh INPUT_DIR [OUTPUT_DIR] [MAX_ITER]

Arguments:
  INPUT_DIR    Directory containing one JSON file per conversation.
  OUTPUT_DIR   Base directory for evaluation outputs.
               Default: evaluation/results
  MAX_ITER     Number of evaluator samples per script invocation.
               Default: 3

Outputs:
  OUTPUT_DIR/ctrs/*.json
  OUTPUT_DIR/wai/*.json
  OUTPUT_DIR/panas_after/*.json
  OUTPUT_DIR/panas_before/*.json

Notes:
  - This script runs the existing GPT-4o evaluation scripts as-is.
  - It must be run from a repo checkout with dependencies installed.
  - The evaluator scripts expect the OpenAI API key in .env via GPT_API_KEY.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$1"
OUTPUT_BASE="${2:-$SCRIPT_DIR/results}"
MAX_ITER="${3:-3}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

if ! [[ "$MAX_ITER" =~ ^[0-9]+$ ]] || [[ "$MAX_ITER" -lt 1 ]]; then
  echo "MAX_ITER must be a positive integer, got: $MAX_ITER" >&2
  exit 1
fi

INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_BASE"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"

CTRS_OUT="$OUTPUT_BASE/ctrs"
WAI_OUT="$OUTPUT_BASE/wai"
PANAS_AFTER_OUT="$OUTPUT_BASE/panas_after"
PANAS_BEFORE_OUT="$OUTPUT_BASE/panas_before"
mkdir -p "$CTRS_OUT" "$WAI_OUT" "$PANAS_AFTER_OUT" "$PANAS_BEFORE_OUT"

run_dir_eval() {
  local workdir="$1"
  local script_name="$2"
  local output_dir="$3"

  (
    cd "$workdir"
    python "$script_name" \
      --input_dir "$INPUT_DIR" \
      --output_dir "$output_dir" \
      --max_iter "$MAX_ITER"
  )
}

echo "Running CTRS evaluation..."
run_dir_eval "$SCRIPT_DIR/CTRS" "ctrs-gpt4o.py" "$CTRS_OUT"

echo "Running WAI evaluation..."
run_dir_eval "$SCRIPT_DIR/WAI" "wai-gpt4o.py" "$WAI_OUT"

echo "Running PANAS-after evaluation..."
run_dir_eval "$SCRIPT_DIR/PANAS" "panas_after-gpt4o.py" "$PANAS_AFTER_OUT"

echo "Running PANAS-before evaluation..."
tmp_dir="$(mktemp -d /tmp/reasoncbt-eval.XXXXXX)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

shopt -s nullglob
input_files=("$INPUT_DIR"/*.json)
shopt -u nullglob

if [[ ${#input_files[@]} -eq 0 ]]; then
  echo "No JSON conversation files found in: $INPUT_DIR" >&2
  exit 1
fi

for input_file in "${input_files[@]}"; do
  base_name="$(basename "$input_file")"
  temp_input="$tmp_dir/${base_name%.json}.json"
  temp_output_dir="$tmp_dir/${base_name%.json}_out"
  mkdir -p "$temp_output_dir"

  python - "$input_file" "$temp_input" <<'PY'
import json
import sys

source_path, target_path = sys.argv[1], sys.argv[2]

with open(source_path, "r", encoding="utf-8") as f:
    data = json.load(f)

patient_profile = data.get("patient_profile")
if not isinstance(patient_profile, dict):
    raise SystemExit(f"Conversation file is missing patient_profile: {source_path}")

required_keys = ("personal_info", "more_information", "attitude")
missing = [key for key in required_keys if key not in patient_profile]
if missing:
    raise SystemExit(
        f"Conversation file has incomplete patient_profile ({', '.join(missing)}): {source_path}"
    )

with open(target_path, "w", encoding="utf-8") as f:
    json.dump([patient_profile], f, indent=2)
PY

  (
    cd "$SCRIPT_DIR/PANAS"
    python "panas_before-gpt4o.py" \
      --input_file "$temp_input" \
      --output_dir "$temp_output_dir" \
      --max_iter "$MAX_ITER"
  )

  generated_file="$temp_output_dir/simulation_1.json"
  if [[ ! -f "$generated_file" ]]; then
    echo "PANAS-before did not produce the expected file for: $input_file" >&2
    exit 1
  fi

  cp "$generated_file" "$PANAS_BEFORE_OUT/$base_name"
done

echo "Evaluation completed."
echo "CTRS results: $CTRS_OUT"
echo "WAI results: $WAI_OUT"
echo "PANAS-after results: $PANAS_AFTER_OUT"
echo "PANAS-before results: $PANAS_BEFORE_OUT"
