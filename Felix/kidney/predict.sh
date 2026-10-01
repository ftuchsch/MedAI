#!/usr/bin/env bash
# Usage: bash predict.sh <input.csv> [output.csv] [additional predict.py options]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash predict.sh <input.csv> [output.csv] [additional predict.py options]" >&2
    exit 1
fi

INPUT="$1"
shift
OUTPUT="predictions.csv"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    OUTPUT="$1"
    shift
fi

# Use the caller's active environment, or an explicit interpreter override.
exec "${MEDAI_PYTHON:-python3}" "$SCRIPT_DIR/predict.py" \
    --data "$INPUT" --out "$OUTPUT" "$@"
