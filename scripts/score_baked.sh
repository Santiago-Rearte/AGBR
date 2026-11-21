#!/usr/bin/env bash
# Like sample.sh, but the data is baked into the image (no bind mounts), for when
# Docker Desktop file sharing misbehaves.  Usage: scripts/score_baked.sh <image> <runs> <answers.json>
set -u
cd "$(dirname "$0")/.."
IMAGE="${1:-br-score}"
RUNS="${2:-3}"
ANSWERS="${3:-blast-radius-sample/sample/answers.json}"
DATA="${4:-blast-radius-sample/sample}"
export MSYS_NO_PATHCONV=1
mkdir -p .work
for i in $(seq 1 "$RUNS"); do
  start=$(date +%s.%N)
  docker run --rm --platform linux/amd64 --network none --cpus 2 \
    --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
    --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
    --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
    "$IMAGE" > ".work/out_$i.jsonl" 2> ".work/err_$i.log"
  rc=$?
  end=$(date +%s.%N)
  cpu=$(grep -oE "usage_usec [0-9]+" ".work/err_$i.log" | awk '{print $2/1000000}')
  sim=$(python blast-radius-sample/score.py "$DATA" "$ANSWERS" ".work/out_$i.jsonl" | grep -E "^Simulated|^Misses|Rejected" | tr '\n' ' ')
  bom=$(head -c 3 ".work/out_$i.jsonl" | od -An -tx1 | tr -d ' ')
  echo "run $i: exit=$rc wall=$(python -c "print(round($end-$start,1))")s cpu=${cpu}s $sim first_bytes=$bom"
done
