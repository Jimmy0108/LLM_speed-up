#!/usr/bin/env bash
# ============================================================
# Gemma 4 E4B Q4_K_M + MTP 一鍵啟動腳本
# 建立日期：2026-11-25
# 硬體：Jetson Orin NX 16GB / MAXN_SUPER
# 效果：decode 37.7 t/s（無 MTP 為 13.1 t/s，約 2.9x）
# ============================================================
set -euo pipefail

LLAMA_SERVER="$HOME/local-llm/build-sm87/bin/llama-server"
BASE=$(find "$HOME/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF" -name "gemma-4-E4B-it-Q4_K_M.gguf" | head -1)
MTP=$(find "$HOME/.cache/huggingface/hub/models--ggml-org--gemma-4-E4B-it-GGUF" -name "mtp-gemma-4-E4B-it-Q4_0.gguf" | head -1)

if [[ -z "$BASE" || -z "$MTP" ]]; then
    echo "錯誤：找不到模型檔案。"
    echo "  BASE=$BASE"
    echo "  MTP=$MTP"
    exit 1
fi

echo "Base : $BASE"
echo "Draft: $MTP"
echo "啟動中（context=4096, MTP n=3, KV q8_0, threads 6）..."

exec "$LLAMA_SERVER" \
    -m "$BASE" \
    -md "$MTP" \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    -ngl 99 -fa on -c 4096 -b 512 -ub 256 -t 6 \
    -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port 8080