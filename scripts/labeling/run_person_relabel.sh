#!/usr/bin/env bash
# Supervises the Kimi PERSON re-scoring run (audit point 3).
#
# Kimi's coding plan has a ~5-hour token quota wall. On quota exhaustion the
# labeler aborts (exit 2); this supervisor waits RESUME_WAIT and resumes from the
# side-file. It gives up only after MAX_STALL_CYCLES consecutive zero-progress
# cycles (account/API dead) so it never spins forever.
set -u
PROJECT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
OUT="$PROJECT/data/labeled/deepseek_t1/person_scores.jsonl"
LOG="$PROJECT/data/labeled/deepseek_t1/person_relabel.log"
WORKERS="${1:-8}"
RESUME_WAIT="${2:-1800}"        # 30 min — wait out a quota window
MAX_STALL_CYCLES="${3:-4}"

cd "$PROJECT" || exit 1
mkdir -p "$(dirname "$OUT")"
count() { [ -f "$OUT" ] && grep -c . "$OUT" 2>/dev/null || echo 0; }

stall=0
while true; do
    before=$(count)
    echo "=== $(date '+%F %T') launch (workers=$WORKERS, scored=$before) ===" >> "$LOG"
    python3 scripts/labeling/relabel_persons_kimi.py --workers "$WORKERS" --output "$OUT" >> "$LOG" 2>&1
    rc=$?
    after=$(count)
    echo "=== $(date '+%F %T') exit rc=$rc  scored $before -> $after ===" >> "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "=== $(date '+%F %T') COMPLETE — all persons scored ===" >> "$LOG"; break
    fi
    if [ "$after" -le "$before" ]; then
        stall=$((stall + 1))
        echo "=== no progress (stall $stall/$MAX_STALL_CYCLES) — quota/API ===" >> "$LOG"
        [ "$stall" -ge "$MAX_STALL_CYCLES" ] && { echo "=== $(date '+%F %T') GIVING UP ===" >> "$LOG"; exit 3; }
    else
        stall=0
    fi
    echo "=== waiting ${RESUME_WAIT}s before resume ===" >> "$LOG"
    sleep "$RESUME_WAIT"
done
