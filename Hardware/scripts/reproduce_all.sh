#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/generated_results}"
export OUTPUT_DIR

cd "${ROOT}"
scripts/run_config1.sh with_bt
scripts/run_config1.sh no_bt
"${PYTHON:-python3}" scripts/check_portable_results.py --results-dir "${OUTPUT_DIR}"
