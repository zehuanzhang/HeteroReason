#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_MODEL="${CONFIG1_TARGET_MODEL:-${TARGET_MODEL:-}}"
exec "$HERE/../common/run_local.sh" Config1 bbeam1 "$HERE/baseline_src" "${1:-smoke}"
