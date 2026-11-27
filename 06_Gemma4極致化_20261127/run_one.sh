#!/bin/bash
# 跑單一 server 配置 + bench，重啟 server
# 用法: ./run_one.sh <tag> <bench_args...> -- <server_args...>
cd "$(dirname "$0")"
LLAMA=~/local-llm/build-sm87/bin/llama-server
BASE=$(find ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF -name "gemma-4-E4B-it-Q4_K_M.gguf" | head -1)
PY=~/Desktop/TensorRT_Edge_LLM_Test/.venv-edgellm/bin/python
TAG="$1"; shift
BENCH_ARGS=()
while [ "$1" != "--" ]; do BENCH_ARGS+=("$1"); shift; done
shift
SERVER_ARGS=("$@")

pkill -x llama-server 2>/dev/null; sleep 3
nohup $LLAMA -m "$BASE" -ngl 99 -fa on -c 4096 -b 512 -ub 256 -t 2 \
  -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8080 "${SERVER_ARGS[@]}" \
  > "server-${TAG}.log" 2>&1 &
for i in $(seq 1 90); do
  sleep 2
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && break
done
H=$(curl -s http://127.0.0.1:8080/health 2>/dev/null)
if [ "$H" != '{"status":"ok"}' ]; then
  echo "server 失敗: $H"; tail -8 "server-${TAG}.log"; pkill -x llama-server 2>/dev/null; exit 1
fi
$PY bench_dual.py --phase P1 --config "$TAG" "${BENCH_ARGS[@]}" 2>&1 | tail -3
pkill -x llama-server 2>/dev/null; sleep 2
echo "done $TAG"