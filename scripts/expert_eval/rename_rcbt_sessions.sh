#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: rename_rcbt_sessions.sh --input DIR

Rename RCBT simulation JSON files in place:
  simulation_0_positive.json -> session_1.json
  simulation_1_neutral.json  -> session_2.json

If DIR contains an rcbt/ subdirectory, files are renamed inside DIR/rcbt.
Otherwise, DIR itself is treated as the rcbt directory.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

input_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      [[ $# -ge 2 ]] || die "--input requires a directory"
      input_dir="$2"
      shift 2
      ;;
    --input=*)
      input_dir="${1#--input=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$input_dir" ]] || die "missing required --input DIR"
[[ -d "$input_dir" ]] || die "input is not a directory: $input_dir"

target_dir="$input_dir"
if [[ -d "$input_dir/rcbt" ]]; then
  target_dir="$input_dir/rcbt"
fi

shopt -s nullglob
files=("$target_dir"/simulation_*.json)
shopt -u nullglob

sources=()
targets=()

for source in "${files[@]}"; do
  name="${source##*/}"
  if [[ "$name" =~ ^simulation_([0-9]+)_[^.]+\.json$ ]]; then
    old_index="${BASH_REMATCH[1]}"
    new_index=$((old_index + 1))
    target="$target_dir/session_${new_index}.json"

    for existing_target in "${targets[@]}"; do
      [[ "$existing_target" != "$target" ]] || die "multiple inputs map to $target"
    done

    [[ ! -e "$target" ]] || die "target already exists: $target"

    sources+=("$source")
    targets+=("$target")
  fi
done

[[ ${#sources[@]} -gt 0 ]] || die "no simulation_<n>_<label>.json files found in $target_dir"

for index in "${!sources[@]}"; do
  mv "${sources[$index]}" "${targets[$index]}"
  echo "renamed ${sources[$index]} -> ${targets[$index]}"
done

echo "Renamed ${#sources[@]} file(s) in $target_dir"
