#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# each child loads its own models, so memory limits this and not cpu cores
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
