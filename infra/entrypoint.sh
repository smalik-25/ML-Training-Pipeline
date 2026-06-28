#!/usr/bin/env bash
# Stage dispatcher for the single-image build. First arg is the stage name; the
# rest pass straight through to the stage's CLI.
#   docker run --rm ml-pipeline ingest --run-date 2025-01-01
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  echo "usage: <stage> [args...]"
  echo "stages: ingest | features | validate | train | register | fixtures"
  exit 0
fi

stage="$1"; shift
case "$stage" in
  ingest)   exec python -m stages.ingest "$@" ;;
  features) exec python -m stages.features "$@" ;;
  validate) exec python -m stages.validate "$@" ;;
  train)    exec python -m stages.train "$@" ;;
  register) exec python -m stages.register "$@" ;;
  fixtures) exec python generate_fixtures.py "$@" ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
