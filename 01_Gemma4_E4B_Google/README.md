# Gemma 4 E4B（4.10B）— 推論加速實驗

> 模型：Google Gemma 4 E4B（4.10B params，42 層，hidden=2560，8Q/2KV，多模態）
> 授權：Apache 2.0（美國）
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER

本目錄收錄同一模型在三種推論方式下的完整數據，**方法 C（Q4_K_M + MTP）為最終冠軍方案**。

---

## 模型來源

| 版本 | 來源 | 大小 | 用途 |
|---|---|---|---|
| 原始 | `google/gemma-4-E4B-it` | — | 參考 |
| INT4 (W4A16) | `hoatac/gemma-4-E4B-it-AWQ`（gs=128, sym=True） | 8.7 GB | 方法 A |
| Q4_0 GGUF | `google/gemma-4-E4B-it-qat-q4_0-gguf` | 4.9 GB | 方法 B |
| Q4_K_M GGUF | `unsloth/gemma-4-E4B-it-GGUF` | 4.64 GB | 方法 C |
| MTP draft | `ggml-org/mtp-gemma-4-E4B-it-Q4_0.gguf` | 57 MB | 方法 C |

---

## 三方法效能對照

| 指標 | 方法A Edge-LLM INT4 | 方法B llama.cpp Q4_0 | 方法C llama.cpp Q4_K_M + MTP |
|---|---|---|---|
| **Decode** | 15.8 tok/s | 15.3 tok/s | **37.7（調參後 40.5）tok/s** 🏆 |
| **TTFT** | **154 ms** | 237 ms | 252 ms |
| **Prefill (pp512)** | 659 | **979** | 857 |
| **RAM** | 15.1 GB（剩0.2） | **8.7 GB** | 10.7 GB |
| 適用 | TTFT 優先 | 省記憶體 | **速度優先** |

> 註：37.7 / 40.5 是**自然思考內容**的 decode；05 公平消融用 `ignore_eos` 強制 64 token 的**任意長文**，Gemma4+MTP 為 18.2 t/s。
> 差異來自 MTP 接受率（思考 ~47–54% vs 任意內容 ~9%），不是量測錯誤。跨模型公平對照見 `../05_全模型消融實驗_20261126/`。

---

## 方法 A：Edge-LLM INT4（TensorRT / AWQ W4A16）

- **路徑**：`方法A_Edge-LLM_INT4/`
- **做法**：TensorRT Edge-LLM 編譯 Gemma 4 E4B INT4（hoatac AWQ），TensorRT runtime 執行。
- **實測**：TTFT 154 ms、Decode 15.8 tok/s、Prefill(512) 659 tok/s、RAM 15.1 GB（剩 0.2 GB ⚠️）。
- **結論**：TTFT 最快（比 llama.cpp 快 35%），但 decode 無優勢，且 15GB 記憶體在 16GB Orin 上幾乎無餘裕，不適合 24/7。
- **檔案**：build/serve log、e2e 原始輸出。

## 方法 B：llama.cpp Q4_0

- **路徑**：`方法B_llama.cpp_Q4_0/`
- **做法**：llama.cpp 跑官方 QAT Q4_0 GGUF，`-ngl 99 -fa on`，KV 預設。
- **實測**：TTFT 237 ms、Decode 15.3 tok/s、**pp512 979 / pp2048 973 / tg128 15.6**（llama-bench）、RAM 8.7 GB。
- **結論**：最省記憶體（8.7GB）、prefill 最快；decode 與 Edge-LLM 相當。
- **檔案**：`gemma4-e4b-llamacpp-bench.log`、`e2e-llamacpp.log`、server log。

## 方法 C：llama.cpp Q4_K_M + MTP ★ 冠軍

- **路徑**：`方法C_llama.cpp_Q4KM_MTP/`
- **做法**：Q4_K_M 主模型 + 57MB MTP draft head，投機解碼 `--spec-draft-n-max`，KV q8_0，ctx 4096。
- **最佳配置（2026-11-26 調參後）**：`--spec-draft-n-max 6 -t 2` → **40.5 tok/s**（原 n-max3/t6 為 36）。
  - 調參完整結果見 `調參測試_20261126/調參測試報告.md`
- **啟動**：
  ```bash
  ./run-gemma4-mtp.sh
  # 或（最佳配置）
  llama-server -m <Q4_K_M> -md <mtp.gguf> --spec-type draft-mtp --spec-draft-n-max 6 \
    -ngl 99 -fa on -c 4096 -b 512 -ub 256 -t 2 -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port 8080
  ```
- **實測**：
  | 指標 | 值 |
  |---|---|
  | Decode（median, 10 runs） | **37.7 tok/s**（調參後最佳 **40.5**） |
  | TTFT | 252 ms（暖機 130–148 ms） |
  | Total（64 tok） | 1927 ms |
  | Draft 接受率 | 49–59%（mean len 2.44–2.77） |
  | RAM | 10.7 GB |
  | 冷啟動首次（長 prompt） | 19.14 tok/s（draft 接受率僅 0.26，冷身） |
- **調參後（2026-11-26）**：最佳 `--spec-draft-n-max 6 -t 2` → 40.5 t/s；threads 越少越好；n-gram 無效；draft Q8_0 無增益。详见 `調參測試_20261126/`。
- **MTP 特性**：對「思考 token」接受率高（~46–62%）、對「最終精簡答案」低（~0–13%）→ 短輸出場景 MTP 幫助有限，這就是速度有時 40、有時 11 的原因。
- **參數敏感度**：
  - `-c 4096` → 37.7 tok/s；`-c 512` → 21.5 tok/s（接受率掉到 17%）。**context 不可縮小**。
  - n-max 3 vs 4 差異小，3 較省。
- **加速原理**：draft head 先猜 3 個，GPU 一次 forward 驗證，平均接受 2.5 個 → 每次 forward 產出 2.5 token 而非 1。
- **檔案**：`原始數據/`（MTP 啟動、冷/暖機、37.7 實測、無 MTP 對照、ctx512、10 次原始數據）、`run-gemma4-mtp.sh`。

---

## 與 CSDN 文章對照

文章 Gemma 2 9B Q4_K_M（同 ctx4096/threads6/KV q8_0，無 MTP）= 16.8 tok/s。
我們**相同參數** + MTP = 37.7 tok/s（**2.24x 文章**），關鍵差異僅在 MTP。

---

## 相關報告（本目錄上層）
- `TensorRT_Edge_LLM_Test_完整報告.md` — 全模型總覽
- `TensorRT_Edge_LLM_Test_方案B_Gemma4_E4B_MTP_報告.md` — MTP 專題
- `gemma4-comparison.md` — 三方比較
- `e2e-comparison.md` — Edge-LLM vs llama.cpp e2e
- `../../05_全模型消融實驗_20261126/全模型消融實驗報告.md` — 6 配置公平消融（MTP on/off × 正確率 × 功耗）
