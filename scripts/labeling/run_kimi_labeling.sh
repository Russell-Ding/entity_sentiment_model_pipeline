#!/usr/bin/env bash
# Supervises the Kimi T1 labeling run across the 5-hour quota windows.
#
# The labeler exits 0 when fully done, or exit 2 when it aborts (quota wall /
# API stall). On exit 2 the supervisor waits, then resumes (the labeler is
# idempotent — it skips already-labeled ids). It gives up only if 3 consecutive
# cycles make ZERO progress (account/API genuinely dead).
set -u

PROJECT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
OUT="$PROJECT/data/labeled/kimi_t1/labels_10ticker.jsonl"
LOG="$PROJECT/data/labeled/kimi_t1/kimi_label.log"
WORKERS="${1:-16}"
RESUME_WAIT="${2:-1800}"     # seconds to wait between quota-window retries (default 30m)
# The Kimi quota refreshes ~every 5h, during which every resume cycle makes ZERO
# progress. Must tolerate a full window (plus margin) before concluding the
# account/API is genuinely dead: 16 cycles * ~35min ≈ 9h of patience.
MAX_STALL_CYCLES="${3:-16}"

cd "$PROJECT" || exit 1
mkdir -p "$(dirname "$OUT")"

count_lines() { [ -f "$OUT" ] && grep -c . "$OUT" 2>/dev/null || echo 0; }

stall=0
while true; do
    before=$(count_lines)
    echo "=== $(date '+%F %T') launch (workers=$WORKERS, labeled=$before) ===" >> "$LOG"
    python3 scripts/labeling/label_news_kimi.py --workers "$WORKERS" --output "$OUT" >> "$LOG" 2>&1
    rc=$?
    after=$(count_lines)
    echo "=== $(date '+%F %T') exit rc=$rc  labeled $before -> $after ===" >> "$LOG"

    if [ "$rc" -eq 0 ]; then
        echo "=== $(date '+%F %T') COMPLETE — all articles labeled ===" >> "$LOG"
        break
    fi

    if [ "$after" -le "$before" ]; then
        stall=$((stall + 1))
        echo "=== no progress this cycle (stall $stall/$MAX_STALL_CYCLES) ===" >> "$LOG"
        if [ "$stall" -ge "$MAX_STALL_CYCLES" ]; then
            echo "=== $(date '+%F %T') GIVING UP after $MAX_STALL_CYCLES stalled cycles ===" >> "$LOG"
            exit 3
        fi
    else
        stall=0   # made progress; reset
    fi
    echo "=== waiting ${RESUME_WAIT}s for quota window before resume ===" >> "$LOG"
    sleep "$RESUME_WAIT"
done
