#!/bin/bash
# Submit training workload to Run:AI
#
# Usage:
#   bash scripts/submit_train.sh <run_id> [extra args...]
#
# Examples:
#   bash scripts/submit_train.sh run01
#   bash scripts/submit_train.sh run02 --max_epochs 50 --lr 5e-4
#   bash scripts/submit_train.sh run01  # auto-resumes from checkpoint

RUN_ID="${1:?Usage: $0 <run_id> [extra train.py args...]}"
shift  # remaining args go to train.py

PROJECT_ROOT="/gpfs0/bgu-rgilad/users/orelgr/deep2"
PYTHON="/gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python"

cd "$PROJECT_ROOT"

echo "=== Training run: $RUN_ID ==="
echo "Extra args: $@"
echo ""

exec $PYTHON -u -m src.train \
    --config configs/base.yaml \
    --run_id "$RUN_ID" \
    "$@"
