#!/bin/sh
# Inference container entrypoint.
# Logs wired env vars, runs startup-summary mode once, then keeps container alive
# so that ad-hoc jobs can be dispatched via: docker compose run --rm inference ...
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

# Run startup-summary mode (scans DATA_DIR, writes startup_summary.json).
# Errors here are non-fatal; the container must remain alive regardless.
python -m src.main || echo "[inference] WARNING: startup summary exited non-zero (continuing)"

echo "[inference] Ready. Keeping container alive for on-demand inference jobs."
exec tail -f /dev/null
