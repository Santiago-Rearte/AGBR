#!/usr/bin/env bash
# Runs the image on the sample set under the scorer's limits and scores it.
# Usage: scripts/sample.sh [image] [extra docker run args...]
set -u
cd "$(dirname "$0")/.."
IMAGE="${1:-br-dev}"
shift || true
SAMPLE="$(pwd)/blast-radius-sample/sample"
OUT=".work/output.jsonl"
ERR=".work/stderr.log"
mkdir -p .work
# Git Bash on Windows rewrites /data-style paths unless told not to.
export MSYS_NO_PATHCONV=1
start=$(date +%s.%N)
docker run --rm --platform linux/amd64 --network none --cpus 2 \
  --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,size=64m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
  --volume "$SAMPLE/commits:/data/commits:ro" \
  --volume "$SAMPLE/tests.json:/data/tests.json:ro" \
  "$@" "$IMAGE" > "$OUT" 2> "$ERR"
rc=$?
end=$(date +%s.%N)
echo "exit=$rc wall=$(python -c "print(round($end-$start,1))")s"
grep -E "^done in|^cgroup" "$ERR"
python blast-radius-sample/score.py blast-radius-sample/sample blast-radius-sample/sample/answers.json "$OUT"
