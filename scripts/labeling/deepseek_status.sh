#!/usr/bin/env bash
P="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
S="$P/data/labeled/deepseek_t1/labels_10ticker.status.json"
LOG="$P/data/labeled/deepseek_t1/deepseek_label.log"
if [ -f "$S" ]; then
  python3 - "$S" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
print(f"[{s.get('updated','?')}] labeled={s.get('labeled_total_on_disk',0)} ok={s.get('ok',0)} "
      f"fail={s.get('failed',0)} rejected={s.get('rejected_high_risk',0)} consec_fail={s.get('consecutive_failures',0)} "
      f"rate={s.get('rate_per_sec',0)}/s eta={s.get('eta_min','?')}m out_tok={s.get('tokens_out',0):,} "
      f"finished={s.get('finished')} abort={s.get('abort_reason')}")
PY
else
  echo "[$(date '+%H:%M:%S')] no status yet (first flush at 50 articles)"
fi
# Terminal markers only matter when the supervisor is actually dead — otherwise we'd
# re-flag a "GIVING UP"/"COMPLETE" line left in the log by a PRIOR cycle while the
# current run is alive. Also only look at the latest cycle (after the last "launch").
if ! pgrep -f run_deepseek_labeling.sh >/dev/null 2>&1; then
  TAIL=$(awk '/=== .* launch /{buf=""} {buf=buf"\n"$0} END{print buf}' "$LOG" 2>/dev/null)
  echo "$TAIL" | grep -q "COMPLETE — all articles" && echo "DS_RUN_COMPLETE"
  echo "$TAIL" | grep -q "GIVING UP" && echo "DS_RUN_GAVE_UP"
  echo "DS_SUPERVISOR_NOT_RUNNING"
fi
