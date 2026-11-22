#!/usr/bin/env bash
# Build one configuration, flash the application, and capture a run.
#
#   ./tools/build_flash_capture.sh logs/bench_mine.log
#   ./tools/build_flash_capture.sh logs/bench_mine.log 240 /dev/ttyACM0
#   NO_FLASH=1 ./tools/build_flash_capture.sh logs/bench_build_only.log
#
# Flashing writes only the application partition at 0x10000. The bootloader,
# partition table and the model partitions are left alone.
#
# Set IDF_PATH to the ESP-IDF v5.4.4 checkout before running.
set -euo pipefail

LOG=${1:?usage: build_flash_capture.sh <log-path> [timeout-seconds] [port]}
TIMEOUT=${2:-240}
PORT=${3:-/dev/ttyACM0}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT/firmware"

if [ -z "${IDF_PATH:-}" ]; then
    echo "IDF_PATH is not set. Source the ESP-IDF v5.4.4 export.sh first." >&2
    exit 2
fi
# shellcheck disable=SC1090
. "$IDF_PATH/export.sh" >/dev/null 2>&1

echo "== build =="
idf.py build

if [ "${NO_FLASH:-0}" = "1" ]; then
    echo "NO_FLASH=1, stopping after the build"
    exit 0
fi

echo "== flash app only =="
esptool.py --port "$PORT" --baud 921600 --chip esp32s3 write_flash \
    --flash_mode dio --flash_freq 80m --flash_size 16MB \
    0x10000 build/llm_s3_test.bin

echo "== capture =="
mkdir -p "$(dirname "$ROOT/$LOG")"
python3 "$ROOT/tools/capture_until.py" "$PORT" "$TIMEOUT" "$ROOT/$LOG" BENCH_DONE

echo "== done: $LOG =="
