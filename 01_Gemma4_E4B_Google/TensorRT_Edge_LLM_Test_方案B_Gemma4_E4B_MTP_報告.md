# Gemma 4 E4B Q4_K_M + MTP 方案 B 完整報告

> 日期：2026-11-25  
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER  
> JetPack 7.2.2 / CUDA 13.2

---

## 一、配置

| 項目 | 方案 B（MTP） | 方案 A（無 MTP） |
|---|---|---|---|
| Base 模型 | unsloth/gemma-4-E4B-it-Q4_K_M.gguf（4.7GB） | 同 |
| MTP draft | ggml-org/mtp-gemma-4-E4B-it-Q4_0.gguf（57MB） | 無 |
| KV cache | q8_0 | q8_0 |
| Context | 4096 | 4096 |
| Threads | 6 | 6 |
| Batch/UBatch | 512/256 | 512/256 |
| Flash Attention | on | on |
| 功耗 | MAXN_SUPER | MAXN_SUPER |

---

## 二、效能數據

### e2e Server（unique prompt，10 runs，64 max tokens）

| 指標 | 方案 B（MTP） | 方案 A（無 MTP） | 文章（Gemma 2 9B） |
|---|---|---|---|
| **TTFT median** | **252ms** | 248ms | ~1200ms |
| **Decode median** | **37.7 t/s** 🏆 | 13.1 t/s | 16.8 t/s |
| **Total median** | **1927ms** | 5071ms | — |
| Decode min/max | 36.6–38.7 | 12.9–13.2 | — |
| **RAM** | **10.7 GB** | 10.4 GB | 7.6 GB |

### llama-bench 純量測（無 cache）

| 指標 | 方案 A（無 MTP） |
|---|---|
| pp512 | 857.4 t/s |
| tg128 | 13.5 t/s |

### Prefill（unique prompt）

| Input | 方案 B（MTP） |
|---|---|
| ~23 tok | 122 t/s |
| ~256 tok | 434 t/s |
| ~512 tok | 3196 t/s* |

*512 tok 數字含 cache 汙染，僅供參考

---

## 三、與文章比較

| 指標 | 文章 Gemma 2 9B | 方案 B Gemma 4 E4B+MTP | 方案 A Gemma 4 E4B |
|---|---|---|---|
| Decode | 16.8 t/s | **37.7 t/s** 🏆 | 13.1 t/s |
| TTFT | ~1200ms | **252ms** 🏆 | 248ms |
| RAM | **7.6 GB** 🏆 | 10.7 GB | 10.4 GB |
| MTP | 無 | **有（2.9x 加速）** | 無 |
| 模型 | Gemma 2 9B（舊） | Gemma 4 E4B（新，多模態） | 同 |

---

## 四、MTP 加速分析

| 指標 | 無 MTP | MTP n=3 | 加速比 |
|---|---|---|---|
| Decode | 13.1 t/s | **37.7 t/s** | **2.88x** |
| Total (64 tok) | 5071ms | 1927ms | 2.63x |

MTP 接受率極高（推測 >70%），因為 Gemma 4 的 MTP head 是 Google 官方訓練的，與 base model 高度對齊。

---

## 五、記憶體

| 項目 | 方案 B | 方案 A |
|---|---|---|
| Server loaded | 10.7 GB | 10.4 GB |
| 剩餘可用 | 0.3 GB | 1.0 GB |
| MTP draft overhead | +0.3 GB | — |

MTP draft head 只增加 0.3GB（Q4_0 量化，僅 57MB 檔案）。

---

## 六、結論

1. **MTP 是決定性的加速**：13.1 → 37.7 t/s（2.9x），遠超文章的 16.8 t/s
2. **TTFT 極低**：252ms，比文章的 ~1200ms 快 4.8x
3. **記憶體是代價**：10.7 GB（剩 0.3GB），不適合同時跑其他大任務
4. **Gemma 4 E4B 是新的最佳選擇**：比 Ornith-1.5-9B（11.6 t/s）快 3.2x