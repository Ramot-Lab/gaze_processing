#!/bin/bash
# Resilient wrapper for run_preprocessing.py: waits out network-mount drops, retries
# on failure. Usage: bash run_preprocessing_with_resume.sh <threshold_based|model_based> [date_str]
METHOD="$1"
DATE_STR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_OUTER_ITERS=40

find_mount() {
  for c in /Volumes/ramot /Volumes/ramot-1 /Volumes/ramot-2 /Volumes/ramot-3 /Volumes/ramot-4 /Volumes/ramot-5 /Volumes/ramot-6 /Volumes/ramot-7 /Volumes/ramot-8 /Volumes/ramot-9; do
    if ls "$c/Noam_M/Results/Behavior/processing_results" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

total_eligible() {
  MOUNT=$(find_mount)
  python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from run_preprocessing import iter_all_subject_dirs
print(sum(1 for _ in iter_all_subject_dirs('$MOUNT/Noam_M/Results/Behavior')))
" 2>/dev/null
}

check_done_count() {
  MOUNT=$(find_mount)
  DATE_FOR_PATH="${DATE_STR:-$(date +%d_%m_%y)}"
  SUFFIX="threshold"
  if [ "$METHOD" = "model_based" ]; then SUFFIX="model"; fi
  CSV="$MOUNT/Noam_M/SDMT_annotated_gaze_${DATE_FOR_PATH}_${SUFFIX}/processed_participants.csv"
  python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$CSV')
    print(df['participant'].nunique())
except Exception:
    print(0)
" 2>/dev/null
}

TARGET_N=$(total_eligible)
echo "target eligible participants: $TARGET_N"

iter=0
while [ "$iter" -lt "$MAX_OUTER_ITERS" ]; do
  iter=$((iter+1))

  if ! find_mount >/dev/null; then
    echo "MOUNT DROPPED - waiting for reconnect..."
    while ! find_mount >/dev/null; do
      sleep 15
    done
    echo "MOUNT BACK - resuming (found at $(find_mount))"
  fi

  cd "$SCRIPT_DIR" || exit 1
  echo "=== launching run_preprocessing.py $METHOD $DATE_STR (outer iter $iter) ==="
  python3 -u run_preprocessing.py "$METHOD" $DATE_STR

  done_count=$(check_done_count)
  echo "done_count=$done_count / $TARGET_N"
  if [ -n "$TARGET_N" ] && [ "$done_count" -ge "$TARGET_N" ]; then
    echo "ALL PARTICIPANTS DONE - exiting wrapper"
    break
  fi
  sleep 5
done

echo "WRAPPER EXITED"
