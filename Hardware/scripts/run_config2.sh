#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-with_bt}"

case "${MODE}" in
  with_bt)
    CONFIG_FILE="${ROOT}/simulator/configs/config2.json"
    ;;
  *)
    echo "Usage: $0 [with_bt]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/generated_results}"
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

"${PYTHON_BIN}" -m simulator.cli \
  --config "${CONFIG_FILE}" \
  --input-root "${ROOT}/inputs/Config2/with_bt" \
  --output "${OUTPUT_DIR}/config2_${MODE}.json" \
  --csv "${OUTPUT_DIR}/config2_${MODE}.csv"
