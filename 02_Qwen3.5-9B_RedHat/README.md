# Qwen3.5-9B（9B）— 推論實驗（歷史對照）

> 模型：Qwen3.5-9B（**中國阿里巴巴系模型**，僅作歷史對照，非最終方案）
> 來源：`RedHatAI/Qwen3.5-9B-quantized.w4a16`（Apache 2.0，compressed-tensors W4A16, gs=128, sym=True）
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER

> **合規備註**：依需求不使用中國模型作為最終方案，本模型僅保留作為 9B 級別的對照數據。

---

## 方法 A：Edge-LLM INT4

- **路徑**：`方法A_Edge-LLM_INT4/`
- **做法**：TensorRT Edge-LLM 編譯 RedHatAI W4A16 checkpoint（11 GB）。
- **實測**：

| 指標 | 值 |
|---|---|
| TTFT median | 181 ms |
| Decode | 10.0 tok/s |
| Prefill (512 tok) | 509 tok/s |
| RAM（loaded） | 14.7 GB（剩 0.3 GB ⚠️） |

- **結論**：
  - 9B 模型在 16GB UMA 上受頻寬限制，decode 僅 10.0 tok/s，**比 4B 的 15.8 慢 37%**。
  - 14.7 GB 記憶體幾乎吃滿，不適合 24/7。
  - 對照結論：同樣是 9B，Ornith-1.5-9B 在 llama.cpp 上反而更快（12.1 tok/s）且更省記憶體。

---

## 檔案
- `qwen35-9b-int4-first-test.log` — 首次測試
- `e2e-qwen35-9b.log` — e2e 量測
- `serve-qwen35-9b-int4*.log` — server 輸出
- `build-qwen35-9b-int4.log` — 編譯 log
