# Jetson Orin NX 16GB — LLM 推論加速實驗歸檔

> 歸檔日期：2026-11-26
> 硬體：Jetson Orin NX 16GB（統一記憶體 UMA，LPDDR5 ~102.4 GB/s）
> 系統：JetPack 7.2.2 / CUDA 13.2 / TensorRT 11.4.1 + 10.16.2
> 功耗：MAXN_SUPER
> llama.cpp：`~/local-llm/build-sm87`（v1.0.0-dev, commit `2d8a3250`, CUDA SM87）

本倉庫收錄在 Jetson Orin NX 16GB 上，針對多個模型、多種推論框架（llama.cpp / TensorRT Edge-LLM）所做的**完整效能實驗數據與操作紀錄**。

> **歸檔原則**：大類別 = 使用的模型；每個模型底下 = 我們實際使用過的加速方法。
> 模型權重檔本身**不納入**本倉庫（過大），僅在下方說明來源與取得方式。

> **最新（2026-11-26 晚）**：完成 6 配置公平消融（統一 llama.cpp / Q4_K_M / prompt / context，唯一變因 = 模型 + MTP）。
> 完整報告：[05_全模型消融實驗_20261126](05_全模型消融實驗_20261126/全模型消融實驗報告.md)。
> 重點：MTP 讓 decode **+37～91%**、正確率不變；MTP 後 Gemma4 18.2 / Ornith 18.9 t/s（強制 64 token）；
> 導航 e2e（124 題平均）：**Gemma4+MTP 900ms**、Ornith v6+MTP 1259ms。
> 自然思考內容的 Gemma4 decode 可達 **40.5 t/s**（01 調參報告），與強制 64 token 的 18.2 量測條件不同，請勿混用。

> **最新（2026-11-27）**：完成雙目標極致化（[06_Gemma4極致化_20261127](06_Gemma4極致化_20261127/報告/總結.md)）。
> 短答：**lite 規則 + n-max3 + ctx2048 → E2E 545–635ms、EM 88.7%**（原 731ms/82.3%）。
> 長答：**thinking ON → decode 40.8 t/s**（雙路徑長答 44.4）。
> 瓶頸：GPU 利用率 97%、功耗 20.6W、無降頻；50 t/s 需改 kernel（Adaptive draft）。

---

## 一、目錄結構

```
LLM_加速歸檔_20261126/
├── README.md                          ← 本檔（總覽）
├── 00_研究與目標文件/
│   ├── deep-research-report.md        ← 加速技術綜述（GQA/FlashAttn/KV量化/投機解碼）
│   ├── 公開測試支持的合理挑戰目標.txt   ← 50/60 tok/s 可行性數學模型
│   ├── 技術考慮指標.txt                ← 優化技術優先級（S/A/B/C）
│   └── 同設備同模型公開實測調查_20261126.md ← 公開數據對照
├── 01_Gemma4_E4B_Google/              ★ 冠軍方案所在地
│   ├── README.md
│   ├── 方法A_Edge-LLM_INT4/
│   ├── 方法B_llama.cpp_Q4_0/
│   └── 方法C_llama.cpp_Q4KM_MTP/      ← 37.7 tok/s（MTP 加速）
├── 02_Qwen3.5-9B_RedHat/
│   ├── README.md
│   └── 方法A_Edge-LLM_INT4/
├── 03_Qwen3.5-4B/
│   ├── README.md
│   └── 方法A_llama.cpp_Q4KM/
├── 04_Ornith-1.5-9B/
│   ├── README.md
│   ├── 方法A_llama.cpp_Q4KM/
│   ├── 方法B_llama.cpp_Q4KM_MTP/
│   └── 方法C_Edge-LLM_INT4_失敗嘗試/
└── 05_全模型消融實驗_20261126/       ← 6 配置公平消融（MTP on/off、功耗、E2E）
    ├── 全模型消融實驗報告.md
    ├── datasets/                      ← nav_testset / nav_holdout（可重現）
    └── run_ablation.py / decode_test.py
└── 06_Gemma4極致化_20261127/         ← 雙目標極致化（短答 E2E / 長答 decode）
    ├── 報告/（Phase0–5b + 總結）
    ├── raw/（逐請求 JSONL + runsheet + summary）
    ├── bench_dual.py / run_one.sh
    └── 對話紀錄.md
```

---

## 二、使用模型與來源

| 模型 | 來源 / 授權 | 取得方式 | 用途 |
|---|---|---|---|
| **Gemma 4 E4B**（4.10B） | Google, **Apache 2.0**（美國） | HF: `google/gemma-4-E4B-it`（原始）<br>INT4: `hoatac/gemma-4-E4B-it-AWQ`（W4A16, gs=128）<br>Q4_0: `google/gemma-4-E4B-it-qat-q4_0-gguf`<br>Q4_K_M: `unsloth/gemma-4-E4B-it-GGUF`<br>MTP draft: `ggml-org/mtp-gemma-4-E4B-it-Q4_0.gguf` | 主線最佳方案 |
| **Qwen3.5-9B**（9B） | RedHatAI, Apache 2.0（**中國阿里巴巴模型**，僅作歷史對照） | HF: `RedHatAI/Qwen3.5-9B-quantized.w4a16` | 對照組 |
| **Qwen3.5-4B** | 本地 GGUF Q4_K_M（中國阿里巴巴系，僅歷史對照） | `~/local-llm/models/Qwen3.5-4B-Q4_K_M.gguf` | 對照組 |
| **Ornith-1.5-9B**（ornith-9b-v6） | 本地 GGUF Q4_K_M | `~/local-llm/models/ornith-9b-v6/ornith-9b-v6-Q4_K_M.gguf` | 原始目標模型 |

> **合規備註**：依需求不使用中國模型作為最終方案；Qwen 系列僅作為早期對照數據保留，**最終採用 Google Gemma 4 E4B**。

### 模型存放位置（不納入本倉庫，僅記錄）
- Gemma 4：`~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/...`、`models--google--gemma-4-E4B-it-qat-q4_0-gguf/...`、`models--ggml-org--gemma-4-E4B-it-GGUF/...`（symlink → blobs，重開機不消失）
- Ornith / Qwen3.5-4B：`~/local-llm/models/`
- TensorRT 引擎：`~/Desktop/TensorRT_Edge_LLM_Test/engines/`（Gemma 2.6G、Ornith 4.3–5.6G、Qwen 系）

---

## 三、全部模型效能總表

| 模型 | 框架 / 精度 | TTFT | Decode (tok/s) | Prefill (tok/s) | RAM |
|---|---|---|---|---|---|
| **Gemma 4 E4B** | **llama.cpp Q4_K_M + MTP** 🏆 | **252 ms** | **37.7（調參後 40.5†）** | 857 (pp512) | 10.7 GB |
| Gemma 4 E4B | llama.cpp Q4_0 | 237 ms | 15.3 | 979 (pp512) | 8.7 GB |
| Gemma 4 E4B | Edge-LLM INT4 (W4A16) | 154 ms | 15.8 | 659 | 15.1 GB |
| Qwen3.5-9B | Edge-LLM INT4 | 181 ms | 10.0 | 509 | 14.7 GB |
| Qwen3.5-4B | llama.cpp Q4_K_M | ~300 ms | 15.8 | 650 | 8.5 GB |
| Ornith-1.5-9B | llama.cpp Q4_K_M | ~400 ms | 12.1 | 360 | 10.0 GB |
| Ornith-1.5-9B | llama.cpp Q4_K_M + MTP | 918 ms* | 18.65* | 238–307 | 7.3–7.8 GB |
| Ornith-1.5-9B | Edge-LLM INT4 | ❌ 無法運行 | — | — | — |

\* Ornith MTP 數字來自導航題型實測（≈250 tok 輸入、≈10 tok 輸出），與 Gemma 的量測條件不同，不宜直接橫比。

> 本表為各方法**原始報告**數字，量測條件（輸出長度、是否 thinking、prompt 型態）不盡相同。
> 同條件公平對照請看 **05_全模型消融實驗_20261126**（強制 64 token 的純 decode、同一份導航測試）。
>
> † 40.5 t/s 為「自然思考內容、64 token 生成、n-max=6 / t=2」的最佳穩定值（`01.../調參測試_20261126/`）。
> 05 的公平測試刻意使用**不可預測的任意長文**（ignore_eos 強制 64 token），Gemma4+MTP 僅 18.2 t/s；
> 兩者差異來自 MTP 接受率（思考內容 ~47–54% vs 任意內容 ~9%），不是量測錯誤。

---

## 四、數據可信度說明（重要）

| 數據類型 | 可信度 | 說明 |
|---|---|---|
| `llama-bench`（pp512/tg128） | ✅ 最高 | 純 compute、無 HTTP、無 cache |
| server e2e + unique prompt | ✅ 高 | 每筆 unique，避開 prefix cache |
| 重複 prompt e2e | ❌ 不可用 | llama.cpp prefix cache 會讓數字虛高 2–10x |
| 記憶體「停止後」數據 | ❌ 不可用 | page cache 殘留 |

> ⚠️ 本倉庫部分早期檔案（如 `04_Ornith.../final-report.md`）曾出現 **4829 tok/s prefill** 這類 cache 汙染數字，已於報告中標註為不可用，請以 `llama-bench` 與 unique-prompt e2e 為準。

---

## 五、核心結論（一頁看懂）

1. **MTP 投機解碼是最大加速點**：Gemma 4 E4B 由 13.1 → **37.7 tok/s（2.9x）**，調參後 **40.5 tok/s**；不改模型、不改量化，僅加 57MB draft head。
   - 公平強制 64 token（任意長文）下為 13.3 → **18.2（+37%）**；差異來自 MTP 接受率依內容變動（思考 ~47–54%、任意長文 ~9%）。
2. **與 CSDN 文章對照**：文章 Gemma 2 9B 純調參 16.8 tok/s；我們相同參數 + MTP → 37.7 tok/s（2.24x 文章）。
3. **全模型公平消融（05）**：MTP 讓 decode **+37～91%**、正確率不變、功耗不增；Ornith v6（LoRA）100%/93.3% 最準，Gemma4 未微調 82.3%/70.0%；導航 e2e Gemma4+MTP **900ms**、Ornith v6+MTP 1259ms（124 題平均）。
4. **Edge-LLM 的價值在 TTFT 不在 decode**：INT4 TTFT 154ms 較快，但 decode 與 llama.cpp 相當，且吃 15GB RAM（剩 0.2GB）。
5. **Ornith-1.5-9B 無法上 Edge-LLM**：無 gs=128 + symmetric 的相容 INT4 checkpoint；llama.cpp Q4_K_M + MTP 是可行路線。
6. **導航短答走 raw `/completion`**：e2e 627–731ms，正確答案只需 7–11 token；開 thinking 反而 3.8s 且答案可能被截斷。長推理 / 長說明才開 thinking（40–48 tok/s）。
7. **下一步目標（見 00_研究與目標文件）**：decode 50 tok/s、E2E(128in/32out) < 1s。硬體頻寬天花板約 49.65 tok/s，需靠 MTP 自適應 + Small GEMM + UMA 優化突破。

---

## 六、各模型詳細文件

- [Gemma 4 E4B](01_Gemma4_E4B_Google/README.md) ★
- [Qwen3.5-9B](02_Qwen3.5-9B_RedHat/README.md)
- [Qwen3.5-4B](03_Qwen3.5-4B/README.md)
- [Ornith-1.5-9B](04_Ornith-1.5-9B/README.md)
- [**全模型公平消融（05）**](05_全模型消融實驗_20261126/全模型消融實驗報告.md) ← 最新
- [Gemma 4 導航任務驗證](01_Gemma4_E4B_Google/導航任務驗證_20261126/導航任務驗證報告.md)（EM/F1、thinking ON/OFF、穩定性）

---

## 七、重現環境

```bash
# llama.cpp（SM87 CUDA build）
~/local-llm/build-sm87/bin/llama-server --version

# Gemma 4 E4B + MTP 一鍵啟動
~/Desktop/LLM_加速歸檔_20261126/01_Gemma4_E4B_Google/方法C_llama.cpp_Q4KM_MTP/run-gemma4-mtp.sh
```

Web GUI：啟動後開 `http://127.0.0.1:8080`（llama.cpp 內建 UI）。

---

## 八、下一步測試計劃（已納入 00_研究與目標文件）

### 8.1 目標（來源：公開測試支持的合理挑戰目標.txt）

| 指標 | 目前 | 主要目標 | 進階目標 |
|---|---|---|---|
| MTP Decode | **40.5 tok/s**（思考內容；公平 64 tok 為 18.2） | **50 tok/s** | 60 tok/s |
| Prefill | 857 tok/s | 1200+ | 1200+ |
| TTFT（128 tok） | 130–250 ms | < 200 ms | < 200 ms |
| E2E（128in/32out） | **0.60–0.89 s**（已 <1s） | < 1 s | ~0.66 s |

> 硬體頻寬天花板：102.4 GB/s × 80% ÷ 1.65 GB/token ≈ **49.65 tok/s**。50 幾乎貼天花板，再往上需降低每 token 資料搬運量。

### 8.2 技術優先級（來源：技術考慮指標.txt）

| 優先級 | 技術 | 解決的問題 |
|---|---|---|
| **S** | Selected Vocabulary Head | Draft 詞彙投影成本 |
| **S** | Adaptive Draft Length | 無效的投機運算 |
| **S** | Small GEMM Specialization | Verification 效率 |
| A | GPU-side Acceptance / Sampling | CPU/GPU 同步 |
| A | Fused Dequantization | 反量化及中間資料讀寫 |
| A | CUDA Graph Shape Bucketing | 不同候選長度的 Graph 重用 |
| A | PLE Layout Optimization | Embedding 存取效率 |
| B | Persistent Kernel / L2 Locality / Split-K / Zero Copy | — |
| C | ARM NEON / CPU Affinity | — |

### 8.3 執行順序（務實版 → 進階版）

1. **務實版（調參，免重編）— ✅ 已完成**：`--spec-draft-n-max 6 -t 2` 為最佳穩定配置 → **40.5 tok/s**（原 35.96，+12.6%）。threads 越少越好、n-gram 無效、draft Q8_0 無增益、KV q4_0 不穩。詳見 `01.../調參測試_20261126/`。
2. **進階版（改 llama.cpp）**：優先實作 S 級三項（Selected Vocab Head / Adaptive Draft Length / Small GEMM），搭配 A 級 CUDA Graph Shape Bucketing、Fused Dequant。目標 50–60。

> 詳細數學模型見 `00_研究與目標文件/公開測試支持的合理挑戰目標.txt`。
