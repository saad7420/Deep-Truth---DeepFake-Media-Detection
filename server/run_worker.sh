#!/usr/bin/env bash
# Start a Celery worker for the analysis queue.
#
#   ./run_worker.sh                 two slots, auto-named from the hostname
#   CONCURRENCY=4 ./run_worker.sh   four slots
#   WORKER_NAME=gpu1 ./run_worker.sh
#
# Run it more than once (different WORKER_NAME) to add capacity — Celery
# distributes across every worker attached to the queue, and the API needs no
# knowledge of how many are running.
set -euo pipefail

cd "$(dirname "$0")"

# Each prefork child loads its own copy of the ViT/ViViT checkpoints, so
# concurrency is bounded by RAM, not cores. Two is a safe default on a
# CPU-only box; raise it only after watching memory during a real run.
CONCURRENCY="${CONCURRENCY:-2}"
WORKER_NAME="${WORKER_NAME:-w1}"
LOGLEVEL="${LOGLEVEL:-info}"
QUEUE="${CELERY_ANALYSIS_QUEUE:-analysis}"

echo "Starting worker '${WORKER_NAME}' with ${CONCURRENCY} slots on queue '${QUEUE}'"

exec celery -A app.queue.celery_app worker \
    --loglevel="${LOGLEVEL}" \
    --concurrency="${CONCURRENCY}" \
    --queues="${QUEUE}" \
    --hostname="${WORKER_NAME}@%h" \
    --without-gossip --without-mingle
