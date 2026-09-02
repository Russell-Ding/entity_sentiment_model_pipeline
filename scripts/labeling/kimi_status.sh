#!/usr/bin/env bash
# Print one compact status line for the Kimi labeling run (cheap to read).
PROJECT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
S="$PROJECT/data/labeled/kimi_t1/labels_10ticker.status.json"
LOG="$PROJECT/data/labeled/kimi_t1/kimi_label.log"

if [ -f "$S" ]; then
    python3 - "$S" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
    print(f"[{s.get('updated','?')}] labeled_total={s.get('labeled_total_on_disk',0)} "
          f"ok={s.get('ok',0)} fail={s.get('failed',0)} empty={s.get('empty_entities',0)} "
          f"consec_fail={s.get('consecutive_failures',0)} rate={s.get('rate_per_sec',0)}/s "
          f"eta={s.get('eta_min','?')}m finished={s.get('finished')} abort={s.get('abort_reason')}")
except Exception as e:
    print(f"status read error: {e}")
PY
else
    n=$(grep -c . "$PROJECT/data/labeled/kimi_t1/labels_10ticker.jsonl" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] no status yet (first flush at 50 articles); labeled lines on disk=$n"
fi

# terminal-state markers so the monitor can alert + stop
if grep -q "COMPLETE — all articles labeled" "$LOG" 2>/dev/null; then echo "KIMI_RUN_COMPLETE"; fi
if grep -q "GIVING UP" "$LOG" 2>/dev/null; then echo "KIMI_RUN_GAVE_UP"; fi
if ! pgrep -f run_kimi_labeling.sh >/dev/null 2>&1; then echo "KIMI_SUPERVISOR_NOT_RUNNING"; fi
