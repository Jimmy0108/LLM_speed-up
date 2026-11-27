# 突破 — thinking OFF 長輸出 58 t/s（結構化輸出）

> 日期：2026-11-27
> 模型：Gemma 4 E4B Q4_K_M + MTP Q4_0（不換、不下載）
> 配置：n-max 掃描、thinking OFF、128 tok、5 題×3 = 15 樣本

## 一、核心發現
**讓輸出「可預測」（JSON/清單）能把 thinking-OFF 長答 decode 從 25 → 58 t/s（2.3×），E2E 從 5.3s → 1.9s。**

## 二、數據

| 內容型態 | n-max | decode | E2E(128tok) | 接受率 |
|---|---|---:|---:|---:|
| 自由長文 | 3 | 25.4 t/s | 5349 ms | 0.20 |
| JSON 結構化 | 3 | 44.0 | 2330 ms | 0.73 |
| JSON 結構化 | 4 | 47.8 | 1999 ms | 0.70 |
| JSON 結構化 | 6 | 52.6 | 1852 ms | 0.65 |
| **JSON 結構化** | **8** | **57.3** | 1970 ms | 0.60 |
| **JSON 結構化** | **10** | **58.2** | 1858 ms | 0.54 |

## 三、為什麼
- MTP 接受率決定 L_eff：`decode = L_eff × BW / B_w`。
- 自由長文內容不可預測 → r≈0.20 → L_eff 低 → 25 t/s。
- JSON 結構化內容高度可預測 → r≈0.54–0.73 → L_eff 高 → 57–58 t/s。
- n-max 越高，結構化內容仍能維持高 L_eff（r 雖降但每輪產出多）。

## 四、正確性
- JSON 輸出合法（`json.loads` 通過）。
- 投機 lossless（前面已驗證 byte-identical）。

## 五、修正的結論
- 之前「50 t/s 不可達」是**針對自由長文**；**結構化內容 50+ 可達**（58.2）。
- 真正口訣：**thinking OFF 要快要讓輸出可預測（JSON/清單/表格）+ n-max 8–10**。

## 六、實用配方
```
llama-server -m <Q4_K_M> -md <mtp-Q4_0.gguf> --spec-type draft-mtp \
  --spec-draft-n-max 8 -ngl 99 -fa on -c 4096 -b 512 -ub 256 -t 2 \
  -ctk q8_0 -ctv q8_0
+ prompt 要求 JSON/清單結構化輸出、thinking OFF
```
→ **decode 57 t/s、E2E(128tok) ~1.9s**

## 七、數據
- `raw/thinking_off_breakthrough.json`
- `json_ab.py` / `nothink_ab.py` — harness