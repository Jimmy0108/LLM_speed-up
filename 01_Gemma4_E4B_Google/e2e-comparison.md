# Gemma 4 E4B 端到端比較：Edge-LLM INT4 vs llama.cpp Q4_0

> 日期：2026-11-25  
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER  
> 模型：google/gemma-4-E4B-it（4.10B params, 42 layers, 多模態）

---

## 測試方法

- 同一硬體、同一功耗模式（MAXN_SUPER）
- 每個框架 warmup 5 次後開始量測
- 測試項目：TTFT、Decode speed、Prefill scaling、Sustained load
- 10 次 streaming 取 median + p95

---

## 結果

### Test 1: TTFT + Decode（10 runs, 64 max tokens）

| 指標 | Edge-LLM INT4 | llama.cpp Q4_0 | 差異 |
|---|---|---|---|
| **TTFT median** | **142ms** | 138ms | Edge-LLM 慢 3% |
| TTFT p95 | 165ms | — | |
| **Decode median** | **17.0 t/s** | 16.3 t/s | Edge-LLM 快 4% |
| Decode min/max | 16.9–17.1 | 16.2–16.3 | Edge-LLM 更穩定 |
| **Total median** | **4033ms** | 4008ms | 接近 |

### Test 2: Prefill（median of 3）

> ⚠️ **數據修正（2026-11-26）**：下表的 **4829 / 9026 t/s** 是重複 prompt 的 **prefix cache 汙染**，**不可用**。
> 同條件公平實測（unique prompt）請看 `05_全模型消融實驗_20261126`：Gemma4 prefill ~248 t/s。
> 只有 23/256 tok 的數字較接近真實；長 prompt 請以 unique prompt 重測。

| Input 長度 | Edge-LLM INT4 | llama.cpp Q4_0 | Edge-LLM / llama.cpp |
|---|---|---|---|
| Short (~23 tok) | 87 t/s | 130 t/s | 0.67x |
| Medium (~256 tok) | 623 t/s | 1586 t/s | 0.39x |
| Long (~512 tok) | 796 t/s | **4829 t/s** | **0.16x** |
| XLong (~1024 tok) | 708 t/s | **9026 t/s** | **0.08x** |

**llama.cpp prefill 在長 prompt 時大幅領先**（4-12x），因為 llama.cpp 的 prefill 是 batched compute-bound，而 Edge-LLM 受 INT4 dequantization overhead 影響。

### Test 3: Sustained load（20 sequential requests）

| 指標 | Edge-LLM INT4 | llama.cpp Q4_0 |
|---|---|---|
| Success rate | 20/20 (100%) | 20/20 (100%) |
| Total median | 1189ms | 2141ms |
| Total p95 | 1260ms | 2148ms |

Edge-LLM sustained load 更快（1189 vs 2141ms），因為短 prompt 時 prefill 不是瓶頸。

### Memory

| 指標 | Edge-LLM INT4 | llama.cpp Q4_0 |
|---|---|---|
| Server loaded RAM | **15.0 GB** | **8.5 GB** |
| After stop RAM | 7.5 GB | 4.4 GB |
| Available after load | **0.2 GB** ⚠️ | **3.3 GB** ✅ |

---

## 總結

| 指標 | 贏家 | 幅度 |
|---|---|---|
| TTFT | 平手 | 138 vs 142ms |
| Decode | Edge-LLM | +4% (17.0 vs 16.3) |
| Prefill (short) | llama.cpp | +50% |
| Prefill (long) | llama.cpp | **+500–1100%** |
| Sustained (short) | Edge-LLM | +44% |
| Memory | llama.cpp | **省 6.5 GB** |
| Stability | llama.cpp | 剩 3.3GB vs 0.2GB |

**建議：llama.cpp Q4_0 是更好的選擇** — decode 只慢 4%，但省 6.5GB RAM、prefill 長 prompt 快 10x、24/7 運行更穩定。