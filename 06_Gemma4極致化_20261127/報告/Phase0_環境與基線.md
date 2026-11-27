# Phase 0 — 環境清理與基線（2026-11-27）

> 配置：Gemma 4 E4B Q4_K_M + MTP draft Q4_0，`--spec-draft-n-max 6 -t 2 -c 4096 -fa on -ctk/ctv q8_0`
> 環境：swap 已關（0）、swappiness=0、system-monitor 已停、Chrome 已關、GPU governor=nvhost_podgov（未鎖頻，代表真實情境）

## 一、基線總表

| 場景 | 輸出 | prefill | decode | E2E | TTFT | 接受率 |
|---|---:|---:|---:|---:|---:|---:|
| nav 短答（out 1/4/8） | 1–8 | 410 | 16.2 | **685 ms** | 465 | 26% |
| 長文 自然（raw 128） | 128 | 375 | 26.1 | 5382 ms | 513 | 26% |
| 長文 任意（ignore_eos 128） | 128 | 377 | 26.0 | 5392 ms | 503 | 26% |

## 二、(in,out) 矩陣（中位數）

| in\out | 1 | 4 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|---:|
| 32 | 497ms | 712 | 700 | 1947 | 5409 |
| 128 | 486ms | 709 | 813 | 1962 | 5396 |
| 512 | 186ms | 313 | 193 | 2117 | 7483 |

- **prefill**：in 32/128 → ~390 t/s；in 512 → **~140 t/s（COMPUTE_BOUND，長 prompt 掉 3 倍）**
- **decode**：長輸出 ~26 t/s；短輸出 0（1 token 就 EOS）
- **draft 接受率**：短答 ~26%、長文 ~26%（raw /completion 無 chat template，非思考內容）

## 三、關鍵發現

1. **短答 E2E 685ms**：prefill 465ms（in~200 token）+ decode ~220ms。瓶頸在 **prefill + 固定開銷**。
2. **長 prompt prefill 崩**：in512 掉到 140 t/s，是短 prompt 的 1/3 → COMPUTE_BOUND。
3. **MTP 在 raw 短答幫助有限**：接受率 26%，且短答 out≤8 時 draft 開銷占比高（SYNC_OVERHEAD）。
4. **環境**：power 7.8–9.6W、temp 58–64°C、swap 0、無 thermal。

## 四、與歸檔對照

| 指標 | 歸檔值 | P0 重現 | 差異原因 |
|---|---|---|---|
| nav 短答 E2E | 731 ms | 685 ms | 本次 swap 已關、背景已清 |
| 長文 decode（任意） | 18.2 t/s | 26.0 t/s | 本次用 nav prompt（較可預測）|
| prefill（短） | 440 | 410 | 正常變異 |

## 五、下一步（Phase 1）
- A1 `--spec-draft-p-min` 掃描（短答 E2E）
- A2 MTP off/1/2/3/6
- A3 思考 on/off
- A5 grammar 約束

## 檔案
- `raw/P0_*_runs.jsonl` — 逐請求
- `raw/runsheet.md` — 人看逐條
- `環境快照/env_P0.txt` — 環境