#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_REL_DEFAULT="exp2_3/results"
TARGET_INPUT="${TARGET_REL_DEFAULT}"
DRY_RUN=0
ASSUME_YES=0
KEEP_GITKEEP=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup_exp2_3_results.sh [options]

Options:
  --target <path>       Target directory to clean (default: exp2_3/results)
  --dry-run             Print what would be removed, do not delete
  --yes                 Do not ask for confirmation
  --delete-gitkeep      Also delete .gitkeep inside target
  -h, --help            Show this help

Examples:
  # Preview only
  bash scripts/cleanup_exp2_3_results.sh --dry-run

  # Clean exp2_3/results directly
  bash scripts/cleanup_exp2_3_results.sh --yes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "Error: --target requires a value."; exit 1; }
      TARGET_INPUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --delete-gitkeep)
      KEEP_GITKEEP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${TARGET_INPUT}" = /* ]]; then
  TARGET_DIR="${TARGET_INPUT}"
else
  TARGET_DIR="${REPO_ROOT}/${TARGET_INPUT}"
fi

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Target directory does not exist, nothing to clean:"
  echo "  ${TARGET_DIR}"
  exit 0
fi

case "${TARGET_DIR}" in
  "${REPO_ROOT}"/*) ;;
  *)
    echo "Refuse to clean path outside repo root."
    echo "Repo root: ${REPO_ROOT}"
    echo "Target   : ${TARGET_DIR}"
    exit 1
    ;;
esac

shopt -s dotglob nullglob
TO_DELETE=()
for entry in "${TARGET_DIR}"/*; do
  base="$(basename "${entry}")"
  if [[ "${KEEP_GITKEEP}" -eq 1 && "${base}" == ".gitkeep" ]]; then
    continue
  fi
  TO_DELETE+=("${entry}")
done
shopt -u dotglob nullglob

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
  echo "Nothing to clean in: ${TARGET_DIR}"
  exit 0
fi

echo "Target directory: ${TARGET_DIR}"
echo "Items to remove: ${#TO_DELETE[@]}"
for p in "${TO_DELETE[@]}"; do
  echo "  - ${p}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "[DRY-RUN] No files were deleted."
  exit 0
fi

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  echo
  read -r -p "Type 'yes' to confirm deletion: " confirm
  if [[ "${confirm}" != "yes" ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

for p in "${TO_DELETE[@]}"; do
  rm -rf "${p}"
done

echo "Cleanup completed: ${TARGET_DIR}"
