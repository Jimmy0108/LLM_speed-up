# TensorRT Edge-LLM INT4 AWQ 完整效能比較報告

> 日期：2026-11-24  
> 硬體：Jetson Orin NX 16GB  
> 功耗模式：MAXN_SUPER  
> JetPack 7.2.2 / CUDA 13.2 / TensorRT 10.16.2

---

## 測試模型

| 模型 | 框架 | 精度 | 量化格式 | group_size | 模型大小 |
|---|---|---|---|---|---|
| Qwen3.5-4B | Edge-LLM | FP16 | — | — | 8.8 GB |
| Qwen3.5-4B | Edge-LLM | INT4 AWQ | modelopt W4A16_AWQ | 128 | 3.6 GB |
| Qwen3.5-4B | llama.cpp | Q4_K_M | GGUF | — | 2.9 GB |
| Qwen3.5-9B | Edge-LLM | INT4 W4A16 | compressed-tensors | 128 | 8.5 GB |
| Ornith-1.5-9B | llama.cpp | Q4_K_M | GGUF | — | 5.4 GB |

---

## 效能比較

### Decode 速度（生成 tok/s）

| 模型 | 框架 | 精度 | Decode (tok/s) |
|---|---|---|---|
| Qwen3.5-4B | Edge-LLM INT4 | W4A16 | **20.8** 🏆 |
| Qwen3.5-4B | llama.cpp Q4_K_M (MAXN) | Q4_K_M | 15.8 |
| Qwen3.5-4B | llama.cpp Q4_K_M (40W) | Q4_K_M | 16.2 |
| Qwen3.5-4B | Edge-LLM FP16 | FP16 | 11.4 |
| Qwen3.5-9B | Edge-LLM INT4 | W4A16 | **11.5** |
| Ornith-1.5-9B | llama.cpp Q4_K_M (MAXN) | Q4_K_M | 12.1 |
| Ornith-1.5-9B | llama.cpp Q4_K_M (40W) | Q4_K_M | ~13-15 |

### TTFT（Time To First Token）

| 模型 | 框架 | 精度 | TTFT |
|---|---|---|---|
| Qwen3.5-4B | Edge-LLM INT4 | W4A16 | **127ms** 🏆 |
| Qwen3.5-4B | Edge-LLM FP16 | FP16 | 151ms |
| Qwen3.5-4B | llama.cpp (MAXN) | Q4_K_M | ~300ms |
| Qwen3.5-9B | Edge-LLM INT4 | W4A16 | **157ms** 🏆 |
| Ornith-1.5-9B | llama.cpp (40W) | Q4_K_M | ~400ms |

### Prefill 速度（512 token prompt）

| 模型 | 框架 | 精度 | Prefill (tok/s) |
|---|---|---|---|
| Qwen3.5-4B | Edge-LLM FP16 | FP16 | 840 |
| Qwen3.5-4B | Edge-LLM INT4 | W4A16 | 808 |
| Qwen3.5-4B | llama.cpp (MAXN) | Q4_K_M | 655 |
| Qwen3.5-9B | Edge-LLM INT4 | W4A16 | 605 |
| Ornith-1.5-9B | llama.cpp (MAXN) | Q4_K_M | 360 |

### 記憶體使用

| 模型 | 框架 | 精度 | 總 RAM | 剩餘可用 |
|---|---|---|---|---|
| Qwen3.5-4B | Edge-LLM INT4 | W4A16 | 6.9 GB | 8.3 GB |
| Qwen3.5-4B | Edge-LLM FP16 | FP16 | 12.8 GB | 2.5 GB |
| Qwen3.5-4B | llama.cpp | Q4_K_M | 8.5 GB | 6.9 GB |
| Qwen3.5-9B | Edge-LLM INT4 | W4A16 | 13.0 GB | 1.7 GB |
| Ornith-1.5-9B | llama.cpp | Q4_K_M | 10.0 GB | 5.0 GB |

---

## 關鍵發現

### 1. Edge-LLM INT4 TTFT 大幅領先

- Qwen3.5-4B INT4: **127ms**（比 llama.cpp 快 2.4x）
- Qwen3.5-9B INT4: **157ms**（比 Ornith llama.cpp 快 2.5x）
- 原因：CUDA Graph + TensorRT 優化 kernel + INT4 權重載入更快

### 2. 4B INT4 decode 速度最強

- Qwen3.5-4B INT4: **20.8 tok/s**（比 llama.cpp 快 32%，比 FP16 快 82%）
- INT4 權重減少記憶體頻寬需求，decode 是 memory-bound

### 3. 9B INT4 decode 與 llama.cpp 接近

- Qwen3.5-9B INT4: **11.5 tok/s**
- Ornith-1.5-9B llama.cpp: **12.1 tok/s**
- 9B 模型 decode 受 16GB RAM 限制，記憶體頻寬是瓶頸
- Edge-LLM INT4 的 prefill 仍比 llama.cpp 快 **68%**（605 vs 360 t/s）

### 4. 9B INT4 記憶體壓力大

- Qwen3.5-9B INT4 用了 13GB RAM（剩 1.7GB）
- 模型權重 8.5GB + KV cache + embedding + runtime overhead
- 不適合長時間 24/7 運行，有 OOM 風險

### 5. 4B INT4 是最佳平衡點

- 效能最強（decode 20.8 t/s, TTFT 127ms）
- 記憶體充足（6.9GB，剩 8.3GB）
- 適合 24/7 運行

---

## Ornith-1.5-9B 在 Edge-LLM 的問題

| 嘗試 | 結果 | 原因 |
|---|---|---|
| FP16 直接跑 | ❌ | 17GB > 11GB 可用記憶體 |
| AWQ 量化 (group_size=32) | ❌ build 成功但 runtime 失敗 | v1 plugin 在 SM87 上 execute() 失敗 |
| 找 group_size=128 的 Ornith INT4 | ❌ 無現成版本 | cyankiwi 只有 group_size=32 |
| 用 Qwen3.5-9B 代替 | ✅ 成功 | RedHatAI w4a16 group_size=128 |

**結論：Ornith-1.5-9B 目前無法在 Edge-LLM 上跑。**  
需要 group_size=128 的 Ornith INT4 checkpoint，但市面上沒有。  
`RedHatAI/Qwen3.5-9B-quantized.w4a16` 是目前唯一可行的 9B 替代。

---

## 三方比較總結（9B 模型）

| 指標 | Edge-LLM INT4 (Qwen3.5-9B) | llama.cpp Q4_K_M (Ornith-1.5-9B) | 差異 |
|---|---|---|---|
| TTFT | **157ms** | ~400ms | Edge-LLM 快 2.5x |
| Prefill | **605 t/s** | 360 t/s | Edge-LLM 快 68% |
| Decode | 11.5 t/s | **12.1 t/s** | llama.cpp 快 5% |
| RAM | 13.0 GB | **10.0 GB** | llama.cpp 省 3GB |
| 24/7 穩定 | ⚠️ 剩 1.7GB | ✅ 剩 5GB | llama.cpp 更穩 |

**9B 模型：Edge-LLM 在 TTFT 和 prefill 大幅領先，但 decode 和記憶體效率 llama.cpp 略勝。**