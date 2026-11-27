# C1 — Adaptive Draft Length 研究計畫（Gemma 4 E4B）

> 目標：在**不換模型、不下載**前提下，把思考內容 decode 從 40.8 推向 50+。
> 模型：Gemma 4 E4B Q4_K_M + MTP draft Q4_0。
> 硬體：Orin NX 16GB / MAXN_SUPER / llama.cpp 2d8a3250（SM87）。
> 鎖頻：不推薦，最後手段。

## 一、瓶頸回顧（實測）
- 思考內容 decode 40.8 t/s；理論 L_eff 3.82 × BW 62 / 4.64 = **51 t/s**。
- gap 21% = 每輪 verify/同步固定開銷。
- n-max 4/6/8 都 40–41 → 固定 n 已飽和；**依內容動態調 n** 才是剩餘杠杆。

## 二、核心假說
> 不同內容的最佳 draft 長度不同：思考/推理 r~0.48 適合大 n；短答/任意 r~0.1 適合小 n。
> 固定 n-max=6 對兩者都不最佳。依 rolling 接受率動態調 n，可讓每輪都接近最佳 L_eff。

## 三、注入點（已定位）
| 檔案:行 | 用途 |
|---|---|
| `common/common.h:342` | `common_params_speculative_draft`（加 adaptive 欄位）|
| `common/arg.cpp:4183` | spec 參數 CLI（加 `--spec-adaptive`）|
| `common/speculative.cpp:1526` | MTP class（加 members）|
| `common/speculative.cpp:1959` | draft 迴圈 `params.n_max` 判斷（讀值）|
| `common/speculative.cpp:2012` | accept() 收 n_accepted（更新 rolling r）|

## 四、控制器（方案 A 門檻式 + EWMA）
```
每輪 accept 後：
  r = n_accepted / n_drafted
  r_ema = 0.7*r_ema + 0.3*r        (EWMA)
每輪 draft 前：
  if r_ema < LO (0.25): n_max = max(floor, n_max-1)
  if r_ema > HI (0.55): n_max = min(cap,   n_max+1)
  cap = 原 n_max, floor = max(1, n_min)
```
- 界 [1, cap]，預設 cap=6。
- 投機 lossless → 不影響正確性（輸出與固定 n 一致）。
- 單用戶、per-seq 追蹤。

## 五、實作步驟
1. `common.h`：加 `bool adaptive; float adapt_lo, adapt_hi;`
2. `arg.cpp`：加 `--spec-adaptive`（bool）+ `--spec-adapt-lo/hi`。
3. `speculative.cpp` MTP：加 `adapt_r_ema / adapt_cap / adapt_floor / adapt_drafted[]`；draft 前調 n_max、accept 更新 EMA。
4. 重編 `cmake --build build-sm87 --target llama-server -j4`。
5. A/B：固定 n-max6 vs adaptive，量思考內容 decode、短答 E2E、正確性（貪婪輸出一致）。

## 六、判定
- 思考 decode 提升 ≥5% 且輸出一致 → 成功。
- 若無提升或變慢 → 回溯（門檻/界要調）。
- 正確性閘門：nav EM 不降、貪婪輸出 byte-identical。

## 七、預估
- 思考 40.8 → 43–47（+5–15%）。
- 短答不變或略快（低 r 自動降 n）。
- 若達 47+，再評估 C2（2-row verify）合攻 50。

## 八、風險
- 低（只改 n_max、lossless）。
- 唯一風險：門檻設定不當 → 震盪。用 EWMA + 單步調整緩解。