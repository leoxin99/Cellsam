#!/bin/bash
# ALICE Training Monitor — runs on local machine, queries ALICE via SSH
# Usage: bash scripts/monitor_alice.sh [interval_seconds] [max_checks]
# Default: every 600s (10 min), max 12 checks (2 hours)

INTERVAL=${1:-600}
MAX=${2:-12}
HOST="s3890074@login.alice.universiteitleiden.nl"

for i in $(seq 1 $MAX); do
    echo ""
    echo "===== CHECK $i/$MAX at $(date '+%H:%M:%S') ====="
    ssh $HOST bash -c "'
        squeue -u s3890074 --format=\"%i %P %j %T %M %N\" --noheader 2>/dev/null
        echo \"---L4---\"
        ls -lh ~/CellSam/checkpoints/E_phase1_rebalance_l4_*/*.pt 2>/dev/null
        echo \"---A100---\"
        ls -lh ~/CellSam/checkpoints/E_phase1_rebalance_a100_*/*.pt 2>/dev/null
    '"
    if [ $i -lt $MAX ]; then
        echo "Next check in ${INTERVAL}s..."
        sleep $INTERVAL
    fi
done
echo ""
echo "===== MONITORING COMPLETE ====="
