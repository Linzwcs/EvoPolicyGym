#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Copy all HLBench report directories into one downloadable folder.

Usage:
  ./scripts/collect_html_reports.sh <runs_root> [output_dir]

Example:
  ./scripts/collect_html_reports.sh \
    /mnt/shared-storage-user/liyafu/zhilin/code/hlbench/hlbench/runs \
    /mnt/shared-storage-user/liyafu/zhilin/code/hlbench/hlbench/html_reports

Output layout:
  <output_dir>/<model>/<env>/<run_id>/index.html
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

count=0
while IFS= read -r report_dir; do
  run_dir="${report_dir%/report}"
  rel="${run_dir#"$RUNS_ROOT"/}"
  dest="$OUTPUT_DIR/$rel"
  mkdir -p "$dest"
  cp -R "$report_dir/." "$dest/"
  count=$((count + 1))
  echo "copied: $rel"
done < <(find "$RUNS_ROOT" -type d -name report | sort)

echo "Copied $count report directories into: $OUTPUT_DIR"
