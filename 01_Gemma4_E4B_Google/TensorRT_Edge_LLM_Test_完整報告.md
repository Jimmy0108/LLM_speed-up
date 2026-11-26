# Jetson Orin NX 16GB LLM 推論效能完整報告

> 日期：2026-11-25  
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER  
> JetPack 7.2.2 / CUDA 13.2 / TensorRT 11.4.1 + 10.16.2  
> 功耗模式：MAXN_SUPER（CPU 1984MHz 全核）

---

## 數據可信度說明

| 數據類型 | 可信度 | 原因 |
|---|---|---|
| llama-bench（純 compute） | ✅ 最高 | 無 HTTP overhead、無 cache、5 次重複取平均 |
| Edge-LLM e2e unique prompt | ✅ 高 | 每筆 unique、無 cache 汙染、含 HTTP overhead |
| llama.cpp e2e Run 1（cold） | ⚠️ 中 | 含 HTTP overhead、僅 1 筆樣本 |
| llama.cpp e2e Run 2+ | ❌ 不可用 | prefix cache 汙染，數字虛高 2-10x |
| Memory "after stop" | ❌ 不可用 | page cache 殘留，非 process memory |

**本報告所有 e2e 數據均使用 unique prompt 避免 cache 汙染。**

---

## 一、Gemma 4 E4B（4.10B params）— Edge-LLM INT4 vs llama.cpp Q4_0

### 模型資訊

| 項目 | 值 |
|---|---|
| 來源 | google/gemma-4-E4B-it（Apache 2.0，美國） |
| 架構 | Gemma4ForConditionalGeneration（42 層, hidden=2560, 8Q/2KV） |
| Edge-LLM 版本 | hoatac/gemma-4-E4B-it-AWQ（W4A16, gs=128, sym=True, 8.7GB） |
| llama.cpp 版本 | google/gemma-4-E4B-it-qat-q4_0-gguf（Q4_0, 4.9GB） |

### e2e 比較（server, unique prompt, 含 HTTP overhead）

| 指標 | Edge-LLM INT4 | llama.cpp Q4_0 | 差異 |
|---|---|---|---|
| **TTFT median** | **154ms** | 237ms | Edge-LLM 快 35% |
| TTFT min/max | 151/174ms | 226/349ms | |
| **Decode median** | **15.8 t/s** | 15.3 t/s | Edge-LLM 快 3% |
| Decode min/max | 15.8/15.9 | 15.2/15.3 | |
| **Total median** | **4326ms** | 4363ms | 接近 |
| **RAM（loaded）** | 15.1 GB | **8.7 GB** | llama.cpp 省 6.4 GB |
| **RAM 剩餘** | 0.2 GB ⚠️ | **2.8 GB** ✅ | |

### 純 compute 比較（llama-bench, 無 HTTP overhead）

| 指標 | llama.cpp Q4_0 |
|---|---|
| pp512 | **979 ± 17 t/s** |
| pp2048 | **973 ± 15 t/s** |
| tg128（純 decode） | **15.6 ± 0.1 t/s** |

> Edge-LLM 無等效的 component benchmark；e2e prefill 659 t/s 含 HTTP overhead，純 compute 應更高。

### e2e Prefill（unique prompt, Run 1 cold only）

| Input 長度 | Edge-LLM INT4 | llama.cpp Q4_0（cold） |
|---|---|---|
| ~23 tok | 128 t/s | 121 t/s |
| ~256 tok | 597 t/s | 500 t/s |
| ~512 tok | 659 t/s | 590 t/s |
| ~2048 tok | 659 t/s | 813 t/s |

> Edge-LLM prefill 在 512 tok 後趨於平緩（~659 t/s）；llama.cpp 隨長度增加反而更快（compute-bound）。
> 兩者 e2e 數字均含 HTTP overhead，實際 compute 更高（llama-bench pp512=979 t/s）。

---

## 二、Qwen3.5-9B（9B params）— Edge-LLM INT4

### 模型資訊

| 項目 | 值 |
|---|---|
| 來源 | RedHatAI/Qwen3.5-9B-quantized.w4a16（Apache 2.0） |
| 量化 | compressed-tensors W4A16, gs=128, sym=True |
| 大小 | 11 GB |

### e2e 數據（unique prompt）

| 指標 | Edge-LLM INT4 |
|---|---|
| TTFT median | 181ms |
| Decode | 10.0 t/s |
| Prefill (512 tok) | 509 t/s |
| RAM（loaded） | 14.7 GB（剩 0.3 GB ⚠️） |

---

## 三、Ornith-1.5-9B — Edge-LLM 可行性

### 結論：現階段無法在 Edge-LLM 上運行

| 嘗試 | 結果 | 原因 |
|---|---|---|
| FP16 直接跑 | ❌ | 18GB > 11GB 可用記憶體 |
| 本地 AWQ 量化 | ❌ | FP16 forward pass 需 18GB，16GB Orin OOM |
| cyankiwi AWQ (gs=32) | ❌ | v2 plugin 只支援 gs=128；v1 plugin SM87 runtime 失敗 |
| Sohailhosseini AWQ (gs=128, asym) | ❌ | asymmetric 需 zero_point tensor，未提供；binding 不相容 |

### 可行替代路線

| 方案 | 可行性 | 說明 |
|---|---|---|
| x86 GPU 做 AWQ 量化 → 搬 INT4 到 Orin | ✅ | 需 group_size=128 + symmetric=True |
| 用 Qwen3.5-9B INT4 代替 | ✅ 已驗證 | 同架構、非 Ornith 微調、10.0 t/s |
| 維持 llama.cpp Q4_K_M | ✅ 已驗證 | 12.1 t/s、省 5GB RAM、24/7 穩定 |

---

## 四、所有模型總覽

| 模型 | 框架 | 精度 | TTFT | Decode | Prefill(512) | RAM |
|---|---|---|---|---|---|---|
| Gemma 4 E4B | Edge-LLM INT4 | W4A16 | **154ms** | **15.8** | 659 | 15.1 GB |
| Gemma 4 E4B | llama.cpp Q4_0 | Q4_0 | 237ms | 15.3 | 590* | **8.7 GB** |
| Qwen3.5-9B | Edge-LLM INT4 | W4A16 | 181ms | 10.0 | 509 | 14.7 GB |
| Ornith-1.5-9B | llama.cpp Q4_K_M | Q4_K_M | ~400ms | 12.1 | 360 | 10.0 GB |
| Qwen3.5-4B | llama.cpp Q4_K_M | Q4_K_M | ~300ms | 15.8 | 650 | 8.5 GB |

> *llama.cpp e2e prefill 含 HTTP overhead；純 compute llama-bench=979 t/s

---

## 五、我的看法

### 1. Edge-LLM 的價值在 TTFT，不在 decode

Edge-LLM 的 TTFT（154ms）比 llama.cpp（237ms）快 35%，這對短輸出的場景（如你的邏輯任務 6-8 tokens）有實際意義。但 decode 速度兩者幾乎相同（15.8 vs 15.3 t/s），Edge-LLM 沒有 decode 優勢。

### 2. 記憶體是 Edge-LLM 在 16GB Orin 上的致命弱點

Edge-LLM 的 TensorRT runtime overhead（IExecutionContext + CUDA graph pool + shared execution context）固定吃掉 ~5GB。Gemma 4 E4B 用 15.1GB（剩 0.2GB），Qwen3.5-9B 用 14.7GB（剩 0.3GB），都不適合 24/7 穩定運行。llama.cpp 只用 8.7GB，有充足餘裕。

### 3. 9B 模型在 Edge-LLM 上 decode 反而比 4B 慢很多

Qwen3.5-9B decode 只有 10.0 t/s，比 4B 的 15.8 t/s 慢 37%。這是 9B 模型在 16GB 統一記憶體上的頻寬瓶頸。Ornith-1.5-9B 在 llama.cpp 上能跑 12.1 t/s，反而比 Edge-LLM 的 Qwen3.5-9B 快。

### 4. 對你的場景的建議

你的核心需求是 Ornith-1.5-9B 跑 13-15 t/s。目前 **llama.cpp Q4_K_M 仍是最佳選擇**：
- 12.1 t/s decode（與 Edge-LLM 的 9B 相當或更好）
- 10GB RAM（剩 5GB，24/7 穩定）
- 已驗證 1147 次連續請求 100% 成功
- MTP 投機解碼可再提升

Edge-LLM 要等到有 group_size=128 + symmetric=True 的 Ornith INT4 checkpoint（目前不存在），或在 x86 上自行量化後搬過來。

### 5. 數據可信度

本次所有 e2e 測試均使用 unique prompt 避免 cache 汙染。llama.cpp 的 e2e prefill 數字（590 t/s）比 llama-bench（979 t/s）低 40%，因為含 HTTP overhead；比較時應注意這點。Edge-LLM 沒有 component benchmark，e2e 是唯一可比較的數字。