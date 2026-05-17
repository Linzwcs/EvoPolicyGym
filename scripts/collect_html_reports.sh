#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Copy all HLBench report directories and policy versions into one downloadable folder.

Usage:
  ./scripts/collect_html_reports.sh <runs_root> [output_dir]

Example:
  ./scripts/collect_html_reports.sh \
    /mnt/shared-storage-user/liyafu/zhilin/code/hlbench/hlbench/runs \
    /mnt/shared-storage-user/liyafu/zhilin/code/hlbench/hlbench/html_reports

Output layout:
  <output_dir>/<model>/<env>/<run_id>/index.html
  <output_dir>/<model>/<env>/<run_id>/policies/epoch_000/input_policy.py
  <output_dir>/<model>/<env>/<run_id>/policies/epoch_000/submission_policy.py
  <output_dir>/<model>/<env>/<run_id>/policies/final_workspace_policy.py
  <output_dir>/<model>/<env>/<run_id>/policy_manifest.tsv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

RUNS_ROOT="$1"
OUTPUT_DIR="${2:-html_reports_$(date +%Y%m%d-%H%M%S)}"

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "runs_root does not exist or is not a directory: $RUNS_ROOT" >&2
  exit 2
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "output_dir already exists: $OUTPUT_DIR" >&2
  echo "Choose a new output path or remove it explicitly." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    printf ""
  fi
}

copy_policy() {
  local source="$1"
  local dest="$2"
  local manifest="$3"
  local kind="$4"
  local epoch="$5"
  if [[ ! -f "$source" ]]; then
    return
  fi
  mkdir -p "$(dirname "$dest")"
  cp "$source" "$dest"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$kind" "$epoch" "$(file_sha256 "$source")" "$source" "$dest" >>"$manifest"
}

count=0
policy_count=0
while IFS= read -r report_dir; do
  run_dir="${report_dir%/report}"
  rel="${run_dir#"$RUNS_ROOT"/}"
  dest="$OUTPUT_DIR/$rel"
  mkdir -p "$dest"
  cp -R "$report_dir/." "$dest/"

  manifest="$dest/policy_manifest.tsv"
  printf "kind\tepoch\tsha256\tsource\tcopied_to\n" >"$manifest"
  before_count="$policy_count"
  while IFS= read -r epoch_dir; do
    epoch="$(basename "$epoch_dir")"
    copy_policy "$epoch_dir/input/policy.py" "$dest/policies/$epoch/input_policy.py" "$manifest" "input" "$epoch"
    if [[ -f "$epoch_dir/input/policy.py" ]]; then
      policy_count=$((policy_count + 1))
    fi
    copy_policy "$epoch_dir/submission/policy.py" "$dest/policies/$epoch/submission_policy.py" "$manifest" "submission" "$epoch"
    if [[ -f "$epoch_dir/submission/policy.py" ]]; then
      policy_count=$((policy_count + 1))
    fi
  done < <(find "$run_dir/epochs" -maxdepth 1 -type d -name 'epoch_*' 2>/dev/null | sort)
  copy_policy "$run_dir/workspace/system/policy.py" "$dest/policies/final_workspace_policy.py" "$manifest" "workspace_final" "final"
  if [[ -f "$run_dir/workspace/system/policy.py" ]]; then
    policy_count=$((policy_count + 1))
  fi

  copied_policies=$((policy_count - before_count))
  count=$((count + 1))
  echo "copied: $rel (${copied_policies} policies)"
done < <(find "$RUNS_ROOT" -type d -name report | sort)

echo "Copied $count report directories into: $OUTPUT_DIR"
echo "Copied $policy_count policy files"
