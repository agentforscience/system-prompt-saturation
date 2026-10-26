#!/bin/bash
# Run generation jobs sequentially on one GPU, retrying while memory is unavailable.
# Usage: src/run_gpu_queue.sh <gpu> <jobfile> [holder_ctl]
# jobfile lines: <model> <name> <gpu_gb> [extra args...]
GPU=$1; JOBS=$2; CTL=$3
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TORCH_CPP_LOG_LEVEL=ERROR
while read -r MODEL NAME GB EXTRA; do
  [ -z "$MODEL" ] && continue
  for try in $(seq 1 200); do
    echo "$(date +%T) start $NAME on GPU$GPU try $try"
    if [ -n "$CTL" ]; then H="--holder-ctl $CTL"; else H=""; fi
    CUDA_VISIBLE_DEVICES=$GPU python src/generate.py --model $MODEL --name $NAME --gpu-gb $GB $H $EXTRA >> logs/gen_$NAME.log 2>&1
    rc=$?
    [ -n "$CTL" ] && echo 40 > $CTL
    echo "$(date +%T) $NAME exit $rc"
    if [ $rc -eq 0 ] && grep -q "FINISHED\|0 to do" logs/gen_$NAME.log; then break; fi
    sleep 30
  done
done < "$JOBS"
echo "$(date +%T) QUEUE DONE"
