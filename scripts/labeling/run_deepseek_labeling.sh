#!/usr/bin/env bash
# Supervises the DeepSeek v4-flash T1 labeling run.
#
# DeepSeek is pay-as-you-go with NO 5-hour quota wall, so aborts (exit 2) come only
# from transient API issues or a depleted account balance. The supervisor restarts
# on exit 2 after a short wait, and gives up only after 3 consecutive zero-progress
# cycles (balance exhausted / API down) so it doesn't spin forever on a dead account.
set -u

PROJECT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
OUT="$PROJECT/data/labeled/deepseek_t1/labels_10ticker.jsonl"
LOG="$PROJECT/data/labeled/deepseek_t1/deepseek_label.log"
WORKERS="${1:-16}"
RESUME_WAIT="${2:-120}"      # short — no quota window to wait out
MAX_STALL_CYCLES="${3:-3}"
PROMPT="${4:-$PROJECT/data_label_criteria/relabeling_prompt_v2_decisive.md}"

cd "$PROJECT" || exit 1
mkdir -p "$(dirname "$OUT")"
count_lines() { [ -f "$OUT" ] && grep -c . "$OUT" 2>/dev/null || echo 0; }

stall=0
while true; do
    before=$(count_lines)
    echo "=== $(date '+%F %T') launch (workers=$WORKERS, labeled=$before) ===" >> "$LOG"
    python3 scripts/labeling/label_news_deepseek.py --workers "$WORKERS" --output "$OUT" --prompt "$PROMPT" >> "$LOG" 2>&1
    rc=$?
    after=$(count_lines)
    echo "=== $(date '+%F %T') exit rc=$rc  labeled $before -> $after ===" >> "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "=== $(date '+%F %T') COMPLETE — all articles labeled ===" >> "$LOG"
        break
    fi
    if [ "$after" -le "$before" ]; then
        stall=$((stall + 1))
        echo "=== no progress (stall $stall/$MAX_STALL_CYCLES) — likely balance/API issue ===" >> "$LOG"
        [ "$stall" -ge "$MAX_STALL_CYCLES" ] && { echo "=== $(date '+%F %T') GIVING UP ===" >> "$LOG"; exit 3; }
    else
        stall=0
    fi
    echo "=== waiting ${RESUME_WAIT}s before resume ===" >> "$LOG"
    sleep "$RESUME_WAIT"
done
