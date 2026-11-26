# Ornith-1.5-9B（ornith-9b-v6）— 推論加速實驗

> 模型：Ornith-1.5-9B（ornith-9b-v6）
> 來源：本地 GGUF `~/local-llm/models/ornith-9b-v6/ornith-9b-v6-Q4_K_M.gguf`（5.43 GB，Q4_K_M，33 層含 MTP）
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER
> llama.cpp：`~/local-llm/build-sm87`（commit `2d8a3250`，已含 MTP 與 qwen35 融合張量）

本目錄收錄 Ornith-1.5-9B 在三種方式下的數據：llama.cpp Q4_K_M、llama.cpp + MTP、以及 Edge-LLM INT4（失敗）。

---

## 方法 A：llama.cpp Q4_K_M（無 MTP）

- **路徑**：`方法A_llama.cpp_Q4KM/`
- **做法**：llama.cpp 跑 Q4_K_M，`-ngl 99 -fa on`。
- **實測**：

| 指標 | 值 |
|---|---|
| TTFT | ~400 ms |
| Decode | 12.1 tok/s（完整報告）／10.26 tok/s（v6 導航實測，無 MTP） |
| Prefill | 360 tok/s（e2e）／**323 pp512**（llama-bench v6） |
| tg128 | 10.87 tok/s |
| RAM | 10.0 GB（完整報告）／7.3–7.8 GB（v6） |

- **結論**：穩定、省記憶體、24/7 可跑（已驗證 1147 次連續請求 100% 成功）。

## 方法 B：llama.cpp Q4_K_M + MTP

- **路徑**：`方法B_llama.cpp_Q4KM_MTP/`
- **做法**：`--spec-type draft-mtp --spec-draft-n-max 3`。**此模型的 MTP 是內建在權重內**（`blk.32.nextn.*`），不需額外 draft GGUF。
- **實測**（導航題型：≈250 tok 輸入、≈10 tok 輸出）：

| 指標 | 無 MTP | MTP n=3 |
|---|---|---|
| Decode（真實輸出中位數） | 10.26 tok/s | **18.65 tok/s**（≈1.8x） |
| Draft 接受率 | — | 69–76% |
| mean accepted length | — | 3.07–3.26（上限 3） |
| E2E 中位數 | 1761 ms | **1346 ms**（快 23%） |
| 124 題平均 | 1628 ms/題 | 1243 ms/題 |
| 正確率（EM） | 1.0000 | 1.0000（無變化） |

- **注意**：MTP 只加速 decode，對 prefill（TTFT 918ms）幾乎無影響；E2E 受 prefill 拖累，故只快 23%。
- **檔案**：`bench_ornith_v6.py`（冷啟動/prefill/decode/e2e 測試腳本）、bench log、逐題結果 JSON。

## 方法 C：Edge-LLM INT4（失敗嘗試）

- **路徑**：`方法C_Edge-LLM_INT4_失敗嘗試/`
- **結論：現階段無法在 Edge-LLM 上運行**，所有嘗試均失敗：

| 嘗試 | 結果 | 原因 |
|---|---|---|
| FP16 直接跑 | ❌ | 18GB > 11GB 可用記憶體 |
| 本地 AWQ 量化 | ❌ | FP16 forward 需 18GB，16GB Orin OOM |
| cyankiwi AWQ (gs=32) | ❌ | v2 只支援 gs=128；v1 SM87 runtime 失敗 |
| Sohailhosseini AWQ (gs=128, asym) | ❌ | asymmetric 缺 zero_point tensor |
| Sohailhosseini AWQ runtime | ❌ | `linear_attn.in_proj_qkv.qweight` binding 不相容 |

- **可行替代**：x86 GPU 上做 AWQ（gs=128, sym=True）→ 搬 INT4 到 Orin；或維持 llama.cpp。

---

## 相關報告
- `ornith-v6-deploy-report.md` — 部署 + 效能完整報告
- `ornith-v6-memory-analysis.md` — 記憶體分析
- `final-report.md`、`comparison-report*.md` — 跨模型比較（含早期 cache 汙染數字，已標註）
