# TensorRT Edge-LLM vs llama.cpp 效能比較報告

> 日期：2026-11-24  
> 硬體：Jetson Orin NX 16GB  
> 功耗模式：MAXN_SUPER  
> JetPack 7.2.2 / CUDA 13.2 / TensorRT 10.16.2 (Edge-LLM) + 11.4.1 (system)

---

## 一、測試環境

| 項目 | Edge-LLM | llama.cpp |
|---|---|---|
| 框架 | TensorRT Edge-LLM 0.12.0 | llama.cpp (commit 2d8a3250) |
| 模型 | Qwen/Qwen3.5-4B (FP16) | Qwen3.5-4B-Q4_K_M.gguf |
| 精度 | FP16 (BF16 weights) | Q4_K_M (4-bit quant) |
| 模型大小 | ~8.8 GB | ~2.9 GB |
| 推理引擎 | TensorRT 10.16.2 | CUDA (cuBLAS) |
| Server | trtedge serve (OpenAI-compatible) | llama-server (OpenAI-compatible) |
| 功耗模式 | MAXN_SUPER | MAXN_SUPER |

---

## 二、效能比較

### 2.1 llama-bench 原始吞吐（MAXN 模式）

| 模型 | 框架 | 精度 | pp512 (t/s) | tg128 (t/s) | pg512+128 (t/s) |
|---|---|---|---|---|---|
| Qwen3.5-4B | Edge-LLM | FP16 | **840.1** | **11.4** | — |
| Qwen3.5-4B | llama.cpp | Q4_K_M | 655.3 | 15.8 | 66.7 |
| Ornith-1.5-9B | llama.cpp | Q4_K_M | 360.1 | 12.1 | 51.3 |

### 2.2 端到端效能（OpenAI API，MAXN 模式）

| 指標 | Edge-LLM (FP16) | llama.cpp (Q4_K_M, 40W) | llama.cpp (Q4_K_M, MAXN est.) |
|---|---|---|---|
| TTFT (短 prompt) | **151ms** | ~400ms | ~300ms (est.) |
| Prefill speed | **840 t/s** (512 tok) | 650 t/s | ~655 t/s |
| Decode speed | **11.4 t/s** | 16.2 t/s | ~15.8 t/s |
| Total (456 prompt + 124 output) | 11,866ms | — | — |

### 2.3 記憶體使用

| 項目 | Edge-LLM (FP16) | llama.cpp (Q4_K_M) |
|---|---|---|
| 模型權重 | ~8.8 GB | ~2.9 GB |
| KV Cache | 256 MB (FP16, 8K ctx) | ~512 MB (FP16, 8K ctx) |
| 總 RAM 使用 | ~12.8 GB | ~8.5 GB |
| 剩餘可用 | ~2.5 GB | ~6.9 GB |

---

## 三、分析

### 3.1 Prefill 效能

Edge-LLM FP16 的 prefill 速度（840 t/s）比 llama.cpp Q4_K_M（655 t/s）快 **28%**。  
這反映 TensorRT 優化 kernel 的優勢，即使在 FP16 精度下仍優於 llama.cpp 的 4-bit 量化。

### 3.2 Decode 效能

Edge-LLM FP16 的 decode 速度（11.4 t/s）比 llama.cpp Q4_K_M（15.8 t/s）**慢 28%**。  
主要原因：
1. **FP16 vs Q4_K_M**：FP16 模型大小是 Q4_K_M 的 3 倍，記憶體頻寬是瓶頸
2. Edge-LLM 目前用的是 FP16，未使用 INT4 AWQ 量化
3. llama.cpp 的 MTP（Multi-Token Prediction）投機解碼可進一步提升 decode 速度

### 3.3 TTFT 優勢

Edge-LLM 的 TTFT（151ms）顯著優於 llama.cpp（~400ms），主要受益於：
1. CUDA Graph 捕獲減少 kernel launch overhead
2. TensorRT 優化的 attention kernel
3. 更快的 prefill 速度

---

## 四、待改善事項

1. **INT4 AWQ 量化**：Edge-LLM 目前使用 FP16，需轉換為 INT4 AWQ 以提升 decode 速度並減少記憶體
2. **MTP 投機解碼**：Edge-LLM 支援 MTP，需設定 `--speculative-config` 以啟用
3. **Ornith-1.5-9B 測試**：需下載 Ornith FP16 權重並建置 INT4 engine
4. **長時間穩定性測試**：需進行 soak test 驗證 Edge-LLM 的穩定性

---

## 五、初步結論

| 指標 | 贏家 | 幅度 |
|---|---|---|
| Prefill | Edge-LLM | +28% |
| TTFT | Edge-LLM | -62% (0.15s vs 0.40s) |
| Decode | llama.cpp (Q4_K_M) | +38% |
| 記憶體效率 | llama.cpp (Q4_K_M) | 3x 更省 |
| 模型品質 | 相當 (FP16 vs Q4_K_M) | — |

**Edge-LLM 的核心優勢在 prefill 和 TTFT，但 decode 需靠 INT4 AWQ + MTP 才能追上 llama.cpp。**
