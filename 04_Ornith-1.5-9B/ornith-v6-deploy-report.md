# Ornith-1.5-9B v6 × llama.cpp：Jetson 部署與效能報告

> 日期：2026-11-25
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER
> 系統：JetPack 7.2.2、CUDA 13.2
> llama.cpp：`~/local-llm/build-sm87`（v1.0.0、commit `2d8a3250`、CUDA SM87）
> 模型：`ornith-9b-v6-Q4_K_M.gguf`（5.43 GB，Q4_K_M，33 層含 MTP）
> **MTP：已啟用**（`--spec-type draft-mtp --spec-draft-n-max 3`）

---

## 一、部署結果

| 項目 | 值 |
|---|---|
| 模型路徑 | `~/local-llm/models/ornith-9b-v6/ornith-9b-v6-Q4_K_M.gguf` |
| systemd 服務 | `ornith-v6.service`（user unit）|
| 狀態 | enabled + active；`Linger=yes`（未登入也會啟動）|
| API | `http://127.0.0.1:8080` |
| GUI | `http://127.0.0.1:8080/` |
| 啟動參數 | `-c 2048 --parallel 1 --temp 0 -ngl 99 -fa on -t 6 -b 512 -ub 256`<br>`--spec-type draft-mtp --spec-draft-n-max 3` |
| 記憶體 | RSS ≈ 7.3 GB，peak 7.8 GB（`MemoryMax=12G`）|
| 舊服務 | `llama-server.service` 已 disable，避免開機搶 8080 |

> 這個 build 已包含 qwen35 融合張量與 MTP 支援（`attn_qkv_gate` / `ssm_beta_alpha` / `blk.32.nextn.*`），
> **不需要**照 README 重編 b12722。
>
> README 說「MTP 要另外的 draft model」在這版已經是舊資訊：
> `--spec-type draft-mtp` 會直接用主模型內的 MTP 層建立 draft context，
> **不需要**額外的 draft GGUF。

---

## 二、GUI 使用方式

瀏覽器開 **http://127.0.0.1:8080/** 即可（已用 Chrome 開啟）。

- 已套用特製 chat template（`~/local-llm/ornith-v6-chat-template.jinja`）：
  在 GUI 輸入一句路線描述，伺服器會自動前置「規則 + 6 個 few-shot」再送給模型，
  回覆就是逗號分隔的地點清單。
- 例：輸入 `從大廳往會議室走，途中經過櫃檯` → 回覆 `大廳,櫃檯,會議室`
- GUI 的多輪對話只取「最後一則 user 訊息」，每次都是獨立抽取。
- 這個 template 是為導航抽取特化的；一般聊天請改回模型原始 template
  （移除 unit 裡的 `--chat-template-file`）。

---

## 三、速度實測

### 3.1 冷啟動（page cache 已用 `posix_fadvise(DONTNEED)` 清掉）

| 指標 | 無 MTP | **MTP n=3** |
|---|---:|---:|
| 啟動 → log `model loaded` | 8.00 s | **8.65 s** |
| 啟動 → `/health` 200 | 8.13 s | **8.76 s** |
| 冷啟動後第一題 TTFT | 970 ms | **987 ms** |
| 冷啟動後第一題總時間 | 1.96 s | **1.43 s** |

> MTP context 初始化多花約 0.6 秒，換來後續 decode 大幅加速。

### 3.2 純 compute（llama-bench，無 HTTP；llama-bench 不支援 MTP）

| 測試 | 結果 |
|---|---|
| pp512（prefill） | **322.99 ± 2.74 tok/s** |
| tg128（decode，無 MTP） | **10.87 ± 0.06 tok/s** |

### 3.3 Prefill 長度掃描（server `/completion`，`cache_prompt=false`，MTP 開啟）

| 實際 prompt tokens | prefill tok/s | 耗時 |
|---:|---:|---:|
| 115 | 237.9 | 483 ms |
| 220 | 269.8 | 815 ms |
| 430 | 278.9 | 1,542 ms |
| 849 | 301.6 | 2,815 ms |
| 1,689 | 306.9 | 5,504 ms |

> MTP 對 prefill 幾乎沒有影響，數字與未開 MTP 時相當。

### 3.4 Decode（MTP 實際效果）

MTP draft 接受率（server log）：

| 指標 | 值 |
|---|---|
| acceptance rate | **69–76%** |
| mean accepted length | **3.07–3.26**（上限 3）|

真實導航輸出（約 10 tokens）的 decode 速度：

| 指標 | 無 MTP | **MTP n=3** |
|---|---:|---:|
| decode 中位數 | 10.26 tok/s | **18.65 tok/s** |
| 範圍 | 10.25–10.27 | 16.26–27.14 |
| 加速比 | — | **≈ 1.8x** |

> 若強制模型連續產生 128 個 token（`ignore_eos`），MTP 只有 13.5 tok/s，
> 因為模型平常 10 個 token 就 EOS，後面的 token 是低機率內容、draft 命中率差。
> 這個數字不代表實際使用情境。

### 3.5 端到端（導航題型：≈250 tok prompt、≈10 tok 輸出、10 題）

| 指標 | 無 MTP | **MTP n=3** |
|---|---:|---:|
| E2E 中位數 | 1,761 ms | **1,346 ms** |
| E2E 範圍 | 1,529–1,938 ms | **1,198–1,362 ms** |
| TTFT 中位數 | 948 ms | **918 ms** |
| 124 題平均 | 1,628 ms/題 | **1,243 ms/題** |
| holdout 30 題平均 | 1,498 ms/題 | **1,224 ms/題** |

> E2E 只快約 23%，因為 918 ms 的 TTFT（prefill）佔了大半；
> 輸出越長，MTP 的效益越明顯。

### 3.6 與 README 的 PC 數據對照

| 指標 | PC（8 執行緒 CPU） | Jetson Orin NX（MTP） |
|---|---|---|
| prefill | ~87 tok/s | **238–307 tok/s** |
| decode | ~11.1 tok/s | **~18.7 tok/s**（真實輸出）|
| E2E | 3.4 s/題 | **1.35 s/題** |

---

## 四、正確率（MTP 開啟後重測）

| 測試集 | 題數 | EM | Set-F1 | 平均 |
|---|---:|---:|---:|---:|
| `nav_testset` | 124 | **1.0000** | 1.0000 | 1.243 s/題 |
| `nav_holdout_v1` | 30 | **0.9333** | 0.9867 | 1.224 s/題 |

與未開 MTP 完全一致；holdout 的 2 題錯誤仍是 **C20 干擾地名**：

```
h20-001  預期 廣場,檔案室       → 實際 洽談室,廣場,檔案室
h20-003  預期 玄關,門廳         → 實際 玄關,門廳,天台
```

投機解碼沒有造成正確率變化。

---

## 五、日常操作

```bash
# 狀態 / 啟停
systemctl --user status ornith-v6
systemctl --user restart ornith-v6
systemctl --user stop ornith-v6
systemctl --user start ornith-v6

# 即時 log（可看到 draft acceptance）
journalctl --user-unit ornith-v6 -f

# 健康檢查
curl -s http://127.0.0.1:8080/health
```

### 切換 MTP 開關

- 關閉：移除 unit 裡的 `--spec-type draft-mtp --spec-draft-n-max 3` 後 `daemon-reload` + `restart`。
- 調整：`--spec-draft-n-max` 可改 2～5；本機掃描結果 n=3 與 n=4 幾乎相同，n=3 較省。

### API 呼叫（務必用 `/completion`）

```bash
curl -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"<build_raw_prompt 產生的完整 prompt>","n_predict":64,
       "temperature":0,"stop":["\n"],"cache_prompt":false}'
```

⚠️ `/v1/chat/completions` 只有在套用本服務的特製 template 時才好用；
若你的程式直接打 chat endpoint 又用自己的 system prompt，會被蓋掉。
正式整合建議用 `eval_jetson.py` 的 `build_raw_prompt()` 產生 prompt 後打 `/completion`。

### 重跑評估

```bash
cd ~/Desktop/for_jetson
python3 eval_jetson.py --port 8080 --testset datasets/nav_testset.jsonl --tag check
python3 eval_jetson.py --port 8080 --testset datasets/nav_holdout_v1.jsonl --tag check-holdout
```

### 重跑效能測試

```bash
cd ~/Desktop/TensorRT_Edge_LLM_Test/benchmarks
python3 bench_ornith_v6.py     # 會重啟服務量冷啟動，GUI 會斷線約 9 秒
```

---

## 六、已知限制與注意事項

| 項目 | 說明 |
|---|---|
| MTP | 已啟用；使用主模型內嵌的 `blk.32.nextn.*`，無需額外 draft 檔 |
| GUI 特化 | chat template 固定用中文規則 + 6 shots；英文輸入仍走中文規則 |
| C20 干擾題 | holdout 2/30 已知錯誤 |
| 只綁 127.0.0.1 | 不對外開放；要給其他機器用請改 `--host 0.0.0.0` 並加 `--api-key` |
| 記憶體 | 約 7.3–7.8 GB；同時跑其他大模型或大量 Chrome 分頁會吃緊 |
| 桌面重複模型 | `~/Desktop/for_jetson/model/` 還有一份 5.43 GB，可刪（確認後）|

---

## 七、本次產出檔案

| 路徑 | 內容 |
|---|---|
| `benchmarks/ornith-v6-deploy-report.md` | 本報告 |
| `benchmarks/bench_ornith_v6.py` | 冷啟動 / prefill / decode / e2e 測試腳本 |
| `benchmarks/ornith-v6-bench-results.json` | MTP 版完整量測原始數據 |
| `benchmarks/ornith-v6-bench-mtp.log` | MTP 量測 console 輸出 |
| `benchmarks/ornith-v6-bench.log` | 未開 MTP 的量測輸出（對照）|
| `benchmarks/ornith-v6-llama-bench.log` | llama-bench 原始輸出 |
| `benchmarks/ornith-v6-eval-testset.log` | 124 題評估（未開 MTP）|
| `benchmarks/ornith-v6-eval-holdout.log` | 30 題 holdout（未開 MTP）|
| `benchmarks/results_ornith-v6-mtp-testset.json` | 124 題逐題結果（MTP）|
| `benchmarks/results_ornith-v6-mtp-holdout.json` | 30 題逐題結果（MTP）|
| `benchmarks/results_ornith-v6-jetson.json` | 124 題逐題結果（未開 MTP）|
| `benchmarks/results_ornith-v6-holdout.json` | 30 題逐題結果（未開 MTP）|
| `~/local-llm/ornith-v6-chat-template.jinja` | GUI 用的 chat template |
| `~/.config/systemd/user/ornith-v6.service` | 部署服務（已含 MTP 參數）|
