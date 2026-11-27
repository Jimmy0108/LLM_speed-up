# Phase 1 — 短輸出 E2E 極小化（2026-11-27）

> 配置基準：Q4_K_M + MTP draft Q4_0。變因見各表。
> 環境：swap=0、bg=1、power 7.7–9.3W、temp 60–63°C。

## 一、MTP 深度掃描（固定 ctx4096）

| 配置 | E2E 中位 | prefill | decode | TTFT | 接受率 |
|---|---:|---:|---:|---:|---:|
| **A4a ctx2048 b1024 t4 n-max3** | **609 ms** 🏆 | 427 | 19.5 | 445 | 34% |
| A2d n-max3 | 641 ms | 419 | 18.1 | 457 | 32% |
| A2c n-max2 | 648 ms | 420 | 17.3 | 457 | 40% |
| A2b n-max1 | 650 ms | 419 | 16.1 | 458 | 57% |
| A1 p-min0.2 n-max3 | 638 ms | 428 | 19.4 | 448 | 40% |
| A2a 無 MTP | 761 ms | 394 | 11.2 | 486 | — |

**結論**：
- MTP **有開就比沒開快**（641 vs 761ms）。
- **n-max3 最佳**（n-max1/2 接受率雖高但每輪產出少）。
- **ctx2048 + b1024 + t4 再快 5%**（609ms）。
- p-min 0.2 無明顯增益。

## 二、Prompt 長度對 E2E 的影響（關鍵）

| prompt 型態 | prompt_n | E2E | decode |
|---|---:|---:|---:|
| full rules + shots | 187 | 751 ms | 19.8 |
| **lite rules** | 62 | **635 ms** | 31.1 |
| 無 rules | 27 | 545 ms | 38.7 |

→ **E2E 幾乎線性取決於 prompt 長度**。刪掉冗長 rules+shots 是最大杠杆。

## 三、正確率對照（nav_testset 124 題）

| 規則模式 | EM | F1 | ms/題 |
|---|---:|---:|---:|
| **lite** | **0.8871** 🏆 | **0.9661** | 716 |
| full（歸檔） | 0.8226 | 0.9447 | 894 |
| grammar 約束 | 0.8387 | 0.9336 | 1282（慢）|

**重大發現**：**lite 規則不只更快，正確率還更高**（EM 82.3%→88.7%）。冗長的 full rules 反而干擾模型。
Grammar 約束則拖慢 2 倍、正確率沒贏 → 不採用。

## 四、短答最佳配方（Phase 1 結論）

```
llama-server -m <Q4_K_M> -md <mtp-Q4_0.gguf> --spec-type draft-mtp \
  --spec-draft-n-max 3 -ngl 99 -fa on -c 2048 -b 1024 -ub 256 -t 4 \
  -ctk q8_0 -ctv q8_0
+ prompt 用 lite rules（62 token）
```
→ **E2E ~609ms（混合 out）、單純短答 ~545–635ms、EM 88.7%**

## 五、與目標差距
- 目標 ≤300ms，目前 609ms。
- 剩餘瓶頸：prefill 445ms（62 token 也要 445ms？不，是 in~200 的 full；lite 62 token 的 TTFT 待單獨量）。
- **下一步**：Phase 2 長輸出；短答若要 <300ms 需再壓 prefill（見 Phase 1 補充）。

## 檔案
- `raw/P1_*_runs.jsonl` / `_summary.json`
- `raw/P1_lite_test.json` / `P1_gram_test.json`（逐題）
- `raw/runsheet.md`