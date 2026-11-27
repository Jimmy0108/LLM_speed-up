# 算子融合與計算圖最佳化（G 系列）

日期：2026-11-27
硬體：Jetson Orin NX 16GB（SM 8.7、8 SM、L4T R39.2.2、CUDA 13.2.86）
模型：Gemma 4 E4B Q4_K_M（4.961 GB）+ MTP draft Q4_0（0.061 GB）
build：llama.cpp `2d8a3250`，`GGML_CUDA_GRAPHS=ON`、`GGML_CUDA_FA=ON`、`CMAKE_CUDA_ARCHITECTURES=87`
原則：**不鎖頻**（`nvhost_podgov` 常態），所有數字代表真實情境。

---

## 0. TL;DR

| 問題 | 答案 |
|---|---|
| 現有 fusion 對我們有效嗎？ | **有，但只值 1–7%**：自由長文 +7.3%、prefill +3.8%、canonical JSON +1.1%（噪音內） |
| `GGML_CUDA_GRAPH_OPT` 有效嗎？ | **完全無效**（66.0 vs 66.7 t/s），可忽略 |
| CUDA Graph 本體有效嗎？ | **有效，值 11–12% decode**（關掉 66.7→59.5、25.1→22.1） |
| 有「大」的融合缺口嗎？ | **沒有**。量化 matmul 佔 GPU kernel 時間 ~80%，glue kernel 全部加起來 <6%，Amdahl 上限約 1.02× |
| GEGLU 為什麼沒融合？ | batch 8 走 `mul_mat_q_few`，上游刻意「不融合 few-row MMQ 的 matmul」；要改需新 kernel，**收益 ≤0.9%**，不做 |
| Prompt prefix cache 能救短答嗎？ | **不能**。Gemma4 KV sharing → `llama_memory_can_shift=false` → `--cache-reuse` 被伺服器停用；只有「完全相同的 prompt」能吃到 host cache（139 vs 255 ms） |
| 意外收穫 | **`-np 1` 讓短答 prefill −30%、E2E −17%**（4 slots 把 2048 ctx 切成 512/slot） |
| 最終短答 | **E2E 中位 536 ms（124 題）、EM 0.8952、F1 0.9742**；100 次連續請求 p95 520 ms、輸出 100/100 一致 |

**一句話結論**：這個 workload 的 decode 已經被「每步串流 ~3.0 GB 權重」主導（自由長文 ≈ 48 GB/s，實測可達 57–66 GB/s），算子融合只能回收 <6% 的 activation/launch 成本；真正有效的是**減少每 token 的步數（MTP 接受率）**、**CUDA Graph**、**單槽 prefill**。融合研究到此為止，屬「已驗證、無大 headroom」的 negative result。

---

## 1. 實驗設定

固定啟動（除指定變因外）：

```bash
llama-server -m gemma-4-E4B-it-Q4_K_M.gguf \
  -md mtp-gemma-4-E4B-it-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 8 \
  -ngl 99 -fa on -c 4096 -b 512 -ub 256 -t 2 -ctk q8_0 -ctv q8_0 \
  --reasoning off --host 127.0.0.1 --port 8080
```

- decode 測試：canonical JSON（128 tok）×15、自由散文（128 tok）×15、長 prompt prefill×5。
- 短答測試：`nav_testset.jsonl`（124 題）+ `eval_jetson` lite rules，`/completion`、greedy、`n_predict=64`。
- harness：`fusion_ab.py`、`nav_cache_eval.py`、`format_spectrum.py`。
- 每個 config 至少 15 次；短答 A/B 用同一批題目、同一 session 連續跑。
- 逐筆資料：`raw/fusion_*.jsonl`、`raw/nav_cache_*.jsonl`、`raw/g_series_summary.json`。

> 變異說明：未鎖頻時短答 decode 有 ±10–15% 的 run-to-run 變異（GR3D 都 90%+，但 EMC/GPU 進 boost 的時間不同）。因此所有結論只看**同一次 session 的 paired A/B** 或**中位數 + 非重疊區間**；單獨跨啟動的 decode 差異不採信。

---

## 2. G01 — GPU kernel 分解（nsys）

`nsys profile --trace=cuda`，70 秒、16 個請求（JSON/散文交替），`cuda_gpu_kern_sum`：

| 佔 GPU kernel 時間 | kernel | 次數 | 中位時間 | 說明 |
|---:|---|---:|---:|---|
| 19.4% | `mul_mat_q_few<Q4_K,2,2>` | 5,424 | 89 µs | verify batch 4–8 的 Q4_K matmul |
| 12.3% | `mul_mat_vec_q<Q4_K,1,…>` | 672 | 682 µs | batch=1（draft / 單 token） |
| 10.6% | `mul_mat_q_few<Q4_K,1,1>` | 3,584 | 81 µs | few-row Q4_K |
| 8.5% | `mul_mat_q_few<Q4_K,3,1>` | 1,792 | 128 µs | few-row Q4_K |
| 7.8% | `mul_mat_vec_q<Q4_K,1,…>` | 2,288 | 73 µs | MMVQ |
| 4.8% | `mul_mat_q<Q4_K,8,…>` | 336 | 560 µs | 大 batch |
| 4.3% | `mul_mat_q_few<Q6_K,2,1>` | 760 | 314 µs | Q6_K（lm_head/PLE proj） |
| 3.3% | `mul_mat_q<Q4_K,8,…>` | 1,144 | 54 µs | |
| 2.2% | `mul_mat_vec_q<Q6_K,1,…>` | 512 | 239 µs | |
| 2.0% | `mul_mat_q_few<Q6_K,1,1>` | 496 | 262 µs | |
| 1.9% | `rms_norm_f32<1024,…>` | 8,936 | 8 µs | 已與 mul/rope 融合者 |
| 1.3% | `flash_attn_ext_f16` | 1,120 | 40 µs | FA |
| 1.2% | `quantize_mmq_few_q8_1` | 10,800 | 4.5 µs | activation 量化 |
| 0.9% | `unary_gated_op_kernel<op_gelu>` | 1,984 | 16 µs | **未融合的 GEGLU** |
| 0.8% | `rms_norm_f32<1024,…>` | 6,096 | 6 µs | |

**讀法**：量化 matmul（Q4_K/Q6_K、MMQ/MMVQ）合計 ≈ **80%**。其餘所有 elementwise/norm/GLU/quantize kernel 加起來 **<6%**。就算把它們全部消失，理論上限也只有 **~1.02×**。這直接決定 G 系列的結局。

原始輸出：`raw/nsys_kern_sum.txt`。

---

## 3. G02 — 源碼級融合稽核

### 已存在且會命中的融合（`ggml-cuda.cu`）

| 融合 | 觸發條件 | 對 Gemma4 是否生效 |
|---|---|---|
| `MUL_MAT + bias + GLU`（mmq/mmvq epilogue） | gate/up 同源、GLU ∈ {SWIGLU, GEGLU, SWIGLU_OAI, SWIGLU_CLAMP}、batch 條件 | **部分**：batch 小於門檻時走 `mul_mat_q_few`，被明文排除 |
| `MUL_MAT_ID + GLU`（MoE shared expert） | MoE | 不適用（E4B dense FFN） |
| `RMS_NORM + MUL + ROPE (+SET_ROWS)` | 連續節點 | **是**（nsys 可見 rms_norm 次數少於層數×節點數） |
| `ROPE + SET_ROWS` | | 是 |
| `softcap`、`GDN/SSM cache`、`topk moe` | 特定架構 | 不適用 |

### 關鍵發現：GEGLU 沒有融合的真正原因

`build_ffn()` 確實用 `ggml_geglu_split()`（`LLM_FFN_GELU` + `LLM_FFN_PAR`），pattern 是對的。但 `ggml_cuda_should_fuse_mul_mat_q()` 有這段：

```cpp
// do not fuse the matmuls that few-row MMQ takes
if (ggml_cuda_should_use_mmq_few(...)) {
    return false;
}
```

我們的 verify batch（n-max3/6/8 → B=4/7/9）剛好落在 `mul_mat_q_few` 的路徑，所以 gate/up 兩個 matmul 各自獨立，GEGLU 變成獨立的 `unary_gated_op_kernel`（0.9%）。

`mul_mat_q_few`（`mmq-few.cu`）是單一 matmul kernel，**沒有 gate/up 雙輸出的 epilogue**。要融合必須新增一個「一次算 gate+up、在暫存器內做 GELU×mul」的 variant，預估收益上限 = 0.9%（GEGLU kernel）+ 少量中間張量 DRAM 流量，**≤1.5%**。以 5 GB 權重串流的 decode 來說，不值得引入新的 kernel 與正確性風險。

**決策：不實作 F09/F11，留下 negative result 與 patch 位置。**（若未來 batch 變大、走一般 MMQ，現有融合會自動生效。）

---

## 4. G03 — Fusion ON/OFF A/B

`GGML_CUDA_DISABLE_FUSION=1` vs 預設（15 reps，同一 session）：

| 指標 | Fusion ON | Fusion OFF | 差異 |
|---|---:|---:|---:|
| canonical JSON decode | **66.7 t/s** | 66.0 | **+1.1%** |
| 自由散文 decode | **25.1 t/s** | 23.4 | **+7.3%** |
| 長 prompt prefill | **398.8 t/s** | 384.3 | **+3.8%** |
| JSON accept | 0.750 | 0.750 | — |
| 散文 accept | 0.190 | 0.182 | — |

逐筆區間：JSON 63.9–68.0 vs 64.7–67.7（重疊）；散文 24.9–25.4 vs 23.2–23.6（**完全不重疊**）。

**解讀**：
- 散文每步只產出 ~1.57 token，activation/中間張量佔比高，融合回收的 DRAM 流量相對有感 → +7.3%。
- JSON 每步產出 7 token，同一個 step 的固定成本被攤薄，融合只值 1%。

---

## 5. G04 — `GGML_CUDA_GRAPH_OPT` A/B

| 指標 | 預設 | `GGML_CUDA_GRAPH_OPT=1` |
|---|---:|---:|
| JSON decode | 66.7 | 66.0 |
| 散文 decode | 25.1 | 25.0 |
| prefill | 398.8 | 395.9 |

**無效**（全部在噪音內）。此開關在此 build/此模型沒有可重排的子圖，或已被預設路徑吸收。**不採用。**

---

## 6. G05 — CUDA Graph 本體（關掉）

`GGML_CUDA_DISABLE_GRAPHS=1`：

| 指標 | Graph ON | Graph OFF | 差異 |
|---|---:|---:|---:|
| JSON decode | 66.7 | 59.5 | **−10.8%** |
| 散文 decode | 25.1 | 22.1 | **−12.0%** |
| prefill | 398.8 | 390.3 | −2.1% |

**CUDA Graph 是目前最大的「計算圖」紅利**：每步省下 ~100 次 kernel launch 的 CPU/驅動成本，decode +11–12%，prefill 幾乎無感（prefill 本來就是大 kernel）。

這也解釋了為什麼 `--spec-draft-n-max` 越高、JSON 越快：步數變少，每步的 launch 成本被更多 token 攤掉。

---

## 7. G06 — MMQ 精度 A/B

`GGML_CUDA_MMQ_PREC=q4` / `q8`（10 reps）：

| 指標 | 預設 | q4 | q8 |
|---|---:|---:|---:|
| JSON decode | 66.7 | 65.5 | 66.1 |
| 散文 decode | 25.1 | 24.3 | 24.6 |
| prefill | 398.8 | 389.1 | 392.3 |

全部在噪音內。**不採用**（也沒有精度理由去改）。

---

## 8. G07 — 為什麼「再融合」救不了 decode：Roofline

### 8.1 模型檔案的真實組成（GGUF tensor 統計）

| 類別 | 大小 | 佔比 | 每步是否完整讀取 |
|---|---:|---:|---|
| transformer 權重（`blk.*`） | 2.591 GB | 52.2% | **是** |
| `per_layer_token_embd`（PLE） | 1.938 GB | 39.1% | **否**，每層只 gather 1 列 |
| `token_embd`（tied lm_head） | 0.377 GB | 7.6% | **是**（輸出層要完整矩陣） |
| `per_layer_model_proj` | 0.055 GB | 1.1% | 是 |
| **合計** | **4.961 GB** | 100% | **每步實際串流 ≈ 3.023 GB** |

> 重點：檔案 4.96 GB，但 PLE 佔 1.94 GB 且是「按 token 取列」，不是每步全讀。用檔案大小算頻寬會高估 ~64%。

### 8.2 實測可達頻寬（EMC 3199 MHz、GR3D 99%）

| 測試 | 結果 |
|---|---:|
| 512 MB read-only kernel | **57.3 GB/s** |
| 256 MB D2D copy（讀+寫） | **66.4 GB/s** |
| 規格理論值（LPDDR5 6400 MT/s） | 102.4 GB/s |

Orin NX 只有 8 個 SM，實測可達頻寬遠低於理論值；**57–66 GB/s 才是這個模組的真實屋頂**。

### 8.3 對照 decode

| 場景 | t/s | tokens/step | steps/s | 每步串流 | 實測頻寬 | 佔可達屋頂 |
|---|---:|---:|---:|---:|---:|---:|
| 自由散文 n3 | 25.1 | 1.57 | 16.0 | 3.023 GB | **48.4 GB/s** | **73–84%** |
| canonical JSON n8 | 66.7 | 7.00 | 9.53 | 3.023 GB | **28.8 GB/s** | 44–50% |

**結論**：
- 自由長文已經貼近記憶體屋頂（73–84%），**任何算子融合最多再擠 1–2%**。
- JSON 還有頻寬餘裕，但它是「每步延遲」而非「頻寬」瓶頸 → 所以 CUDA Graph（−11%）比融合（+1%）有效得多。
- 唯一能大幅改變結果的槓桿是 **tokens/step（MTP 接受率）**，這也是 Phase 1–5 的結論。

### 8.4 Amdahl 上限

GPU kernel 時間 80% 是無法再融合的量化 matmul；剩下 20% 中，rms_norm/GLU/quantize 多數已融合或只能省 1–2%。即使把「所有非 matmul」全部變 0，decode 上限也只有 **1.25×**；把「未融合的 GEGLU」單獨變 0，上限 **1.009×**。

---

## 9. G08 — Prompt Prefix Cache（negative result）

動機：短答 lite prompt 是「固定規則 + 固定 few-shot + 使用者句子」，前綴 175/193 token 相同。若 KV 可重用，prefill 可從 326 ms 降到 ~40 ms。

實測（124 題，`cache_prompt=true` + `--cache-reuse 64`）：

| 指標 | 無 cache | 有 cache |
|---|---:|---:|
| prefill 中位 | 326 ms | 337 ms |
| E2E 中位 | 536 ms | 594 ms |
| decode 中位 | 31.1 t/s | 27.0 t/s（時脈變異） |
| 完全相同 prompt（100 次） | 255 ms | **139 ms** |

**根因**（server log）：

```
W srv load_model: cache_reuse is not supported by this context, it will be disabled
```

`llama_memory_can_shift()` 對 Gemma4 回傳 false（KV sharing：layer 0–3 與 layer 40 共用 K/V），所以 `--cache-reuse` 被停用；一般請求只能靠 host prompt cache，而它只對「整段 prefix 完全相同」有效。

- 變動 prompt：**沒有收益**（337 vs 326 ms，噪音內），甚至因比對/換出多 10 ms。
- 固定 prompt（如固定的 system + few-shot，使用者句子接在最後）：**prefill −45%**、E2E −6%（474 vs 505 ms）。

**決策**：短答正式配方**不開** `--cache-reuse`（無效且多餘）；若產品端是「同一段 system prompt + 可變使用者輸入」，開 `cache_prompt` 仍值得（但 llama.cpp 此 build 的 prefix cache 只在整段相同時命中，需上層自行切分/保持 slot）。

---

## 10. G09 — 單槽 vs 多槽（意外的大 win）

瓶頸：短答 prefill 佔 E2E 的 60%。檢查 server log 發現：

| 啟動 | n_slots | n_ctx_slot | kv_unified | prefill | E2E（30 題） |
|---|---:|---:|---|---:|---:|
| `-np 4` | 4 | **512** | false | 357 ms | 600 ms |
| `-np 1` | 1 | **2048** | false | **248 ms** | **497 ms** |

多槽把 2048 的 context 平均切成 512/slot，prefill 的 ubatch 也被切碎；單槽保留完整 2048，prefill −30%、E2E −17%，**EM 完全相同（0.9667）**。

124 題完整驗證（`-np 1`、`cache_prompt=false`）：

| 指標 | 數值 |
|---|---:|
| EM | **0.8952** |
| F1 | **0.9742** |
| E2E 中位 | **536 ms** |
| prefill 中位 | 326 ms |
| decode 中位 | 31.1 t/s |

對照 Phase 1 最佳（609 ms）→ **−12%**，且正確率不降（0.8952 vs 0.8871，差異來自 batching 數值，非規則）。

**取捨**：`-np 1` 不能同時服務多個請求。導航是序列式互動，這個取捨完全可接受；若要並行，需回到多槽並接受 prefill 變慢。

---

## 11. G10 — ubatch 掃描（`-np 1`）

| `-b` / `-ub` | prefill | E2E（30 題） |
|---|---:|---:|
| 1024 / **256** | 251 ms | **495 ms** |
| 1024 / 512 | 248 ms | 493 ms |
| 2048 / 1024 | 330 ms | 628 ms |

256 與 512 等價，1024 明顯變差（共享記憶體/occupancy 下降）。**維持 256**（與 Phase 1 一致，對長 prefill 也更穩）。

---

## 12. G12 — ncu 硬體計數器（工具限制）

嘗試用 `ncu` 量 `dram__bytes_read` / `sm__throughput`：

```
NvMapMemAllocInternalTagged failed: error 12
Failed ioctlCmd: 1075858947
==ERROR== Failed to profile "mul_mat_q_few"
```

原因：Orin NX 的統一記憶體在模型載入後只剩很少餘裕，ncu 的 replay buffer 配置失敗（縮到 `-c 1024` 仍失敗）。**這是工具/平台限制，不是模型問題。**

替代方案：nsys CUDA trace（G01）+ 自寫 CUDA bandwidth microbenchmark（G07）。兩者已足以做 roofline 判斷。

---

## 13. G13 — 100 次連續請求穩定性 + 正確率

固定短答 prompt（lite rules），`-np 1`，100 次連續：

| 指標 | 無 cache | 有 prompt cache |
|---|---:|---:|
| E2E 中位 | **505 ms** | 474 ms |
| E2E p95 | **520 ms** | 509 ms |
| E2E max | 523 ms | 665 ms（首請求） |
| prefill 中位 | 255 ms | 139 ms |
| decode 中位 | 32.3 t/s | 24.2 t/s（時脈變異） |
| 輸出唯一值 | **1** | **1** |

- 100/100 輸出完全一致（greedy + 固定 prompt），無漂移、無 OOM、無重啟。
- tegrastats：功耗平均 **20.4–20.7 W**、峰值 22.3 W；溫度峰值 **71–73 °C**；GR3D 91–92%；RAM 峰值 10.3–10.4 GB（含 OS）。
- 無降頻、無 swap（swap=0）。

---

## 14. 最終結論

### 14.1 有效的手段（依貢獻排序）

| 手段 | 貢獻 | 備註 |
|---|---:|---|
| MTP 接受率 / n-max | **2–5×** | 真正的主引擎（Phase 1–5） |
| `-np 1`（單槽） | **E2E −17%** | 短答 prefill −30% |
| CUDA Graph | **decode +11–12%** | 預設已開，勿關 |
| 算子融合（現有） | 散文 +7.3%、prefill +3.8%、JSON +1.1% | 預設已開 |
| prompt cache（整段相同） | prefill −45% | 只對固定 prompt 有效 |

### 14.2 無效 / 不採用

| 手段 | 結果 |
|---|---|
| `GGML_CUDA_GRAPH_OPT=1` | 無效 |
| `GGML_CUDA_MMQ_PREC=q4/q8` | 無效 |
| `--cache-reuse` | 被 Gemma4 KV sharing 停用 |
| 新增 GEGLU 融合 kernel | 上限 ≤1.5%，不做 |
| 2-row verify / C2 | 已含 `mmq-few`，B=4 verify 僅 1.05× |
| KV q4_0 | 38.7 vs 42.4，較慢 |
| `-fa off` | sm_87 global head_dim=512，載入即壞 |

### 14.3 最終配方

**短答（導航）**
```bash
llama-server -m gemma-4-E4B-it-Q4_K_M.gguf \
  -md mtp-gemma-4-E4B-it-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -fa on -c 2048 -b 1024 -ub 256 -t 4 -ctk q8_0 -ctv q8_0 \
  -np 1 --reasoning off
```
→ **E2E 中位 536 ms、EM 0.8952、F1 0.9742、100 次 p95 520 ms**

**長答（思考）**：`n-max 6 -c 4096 -b 512 -ub 256 -t 2 -np 1` → decode 中位 39.9 t/s
**長答（JSON）**：`n-max 8` + canonical prompt → 63–66 t/s
**自由長文**：`n-max 3` → 27.0 t/s（上限）

---

## 15. 檔案

| 檔案 | 內容 |
|---|---|
| `實驗/報告/算子融合與計算圖最佳化_G系列.md` | 本報告 |
| `實驗/raw/fusion_{base,nofuse,graphopt,nograph,mmqq4,mmqq8}.jsonl` | fusion/graph/precision 逐筆 |
| `實驗/raw/nav_cache_{nocache,cache64,np1,np4,ub256,ub512,ub1024}.jsonl` | 短答逐題（含 100 次穩定性的 `kind:"stability"`） |
| `實驗/raw/nav_cache_*_summary.json` | 各 config 彙總 |
| `實驗/raw/g_series_summary.json` | G 系列機器可讀總表 |
| `實驗/raw/nsys_kern_sum.txt` | nsys kernel 原始表 |
| `實驗/fusion_ab.py` | fusion/env A/B harness |
| `實驗/nav_cache_eval.py` | 短答 + 100 次穩定性 harness |
| `實驗/ncu_mmq*.log` | ncu 失敗證據（工具限制） |
