# Qwen3.5-4B — 推論實驗（歷史對照）

> 模型：Qwen3.5-4B（**中國阿里巴巴系模型**，僅作歷史對照，非最終方案）
> 來源：本地 GGUF Q4_K_M（`~/local-llm/models/Qwen3.5-4B-Q4_K_M.gguf`，2.9 GB）
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER

> **合規備註**：依需求不使用中國模型作為最終方案，本模型僅保留作為 4B 級別對照。

---

## 方法 A：llama.cpp Q4_K_M

- **路徑**：`方法A_llama.cpp_Q4KM/`
- **做法**：llama.cpp 跑本地 Q4_K_M GGUF，GPU 全層 offload。
- **實測**：

| 指標 | 值 |
|---|---|
| TTFT | ~300 ms |
| Decode | 15.8 tok/s |
| Prefill | 650 tok/s |
| RAM | 8.5 GB |

- **結論**：4B 級別在 llama.cpp 上 decode ~15.8 tok/s、RAM 8.5GB，屬正常。後續 Gemma 4 E4B（4.10B）在同級表現相近，因此最終改採 Gemma 4 E4B（Google，Apache 2.0）。

---

## 檔案
- `qwen35-4b-first-test.log` — 首次測試
- `Qwen3.5-4B-benchmark.json` — llama-bench 原始數據
- `Qwen3.5-4B-benchmark.stderr.log` — llama-bench stderr
- `build-qwen35-4b-int4.log` — （附帶）INT4 編譯嘗試 log
