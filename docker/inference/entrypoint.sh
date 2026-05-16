#!/bin/sh
# Inference container entrypoint.
#
# Two operating modes:
#
#   1. Long-running (compose up, no extra args):
#      Logs env vars, runs startup-summary, then keeps the container alive with
#      tail -f /dev/null so ad-hoc jobs can be dispatched via:
#        docker compose run --rm inference python -m src.main ...
#        docker compose exec inference python -m src.main ...
#
#   2. Pass-through (docker compose run with explicit command args):
#      Skips keep-alive and exec's the provided command directly.
#      This lets CI / smoke tests drive the container without --entrypoint:
#        docker compose run --rm inference python -m src.main
set -eu

echo "[inference] ============================================================"
echo "[inference] Container starting"
echo "[inference] DATA_DIR          = ${DATA_DIR:-<unset>}"
echo "[inference] LOG_DIR           = ${LOG_DIR:-<unset>}"
echo "[inference] INFERENCE_CHECKPOINT = ${INFERENCE_CHECKPOINT:-<unset>}"
echo "[inference] INFERENCE_CONFIG  = ${INFERENCE_CONFIG:-configs/data_pipeline.yml}"
echo "[inference] INFERENCE_DEVICE  = ${INFERENCE_DEVICE:-auto}"
echo "[inference] API_HOST          = ${API_HOST:-api}"
echo "[inference] API_PORT          = ${API_PORT:-8000}"
echo "[inference] ============================================================"

if [ -z "${INFERENCE_CHECKPOINT:-}" ]; then
    echo "[inference] WARNING: INFERENCE_CHECKPOINT is not set."
    echo "[inference]          The container will stay alive but cannot run"
    echo "[inference]          model inference until INFERENCE_CHECKPOINT is provided."
fi

# Pass-through mode: if arguments were provided, execute them directly and exit.
# Used by CI smoke tests and one-off job dispatch.
if [ "$#" -gt 0 ]; then
    echo "[inference] Pass-through mode: executing: $*"
    exec "$@"
fi

# Long-running mode: run startup-summary then keep the container alive.
# Errors in startup-summary are non-fatal; the container must remain alive.
python -m src.main || echo "[inference] WARNING: startup summary exited non-zero (continuing)"

echo "[inference] Ready. Keeping container alive for on-demand inference jobs."
exec tail -f /dev/null
