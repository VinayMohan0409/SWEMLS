#!/usr/bin/env bash
set -euo pipefail

docker build -t coursework3 .
docker build -t simulator -f Dockerfile.simulator .

docker network inspect assess-net >/dev/null 2>&1 || docker network create assess-net

docker run --rm -d \
  --name sim \
  --network assess-net \
  -p 7435:8440 -p 7436:8441 \
  simulator --messages=/data/messages.mllp

trap 'docker stop sim >/dev/null 2>&1 || true' EXIT

docker run --rm \
  --name coursework3 \
  --network assess-net \
  -e MLLP_ADDRESS=sim:8440 \
  -e PAGER_ADDRESS=sim:8441 \
  -v "$PWD:/data" \
  coursework3
