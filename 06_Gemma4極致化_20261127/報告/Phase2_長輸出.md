# Phase 2 — 長輸出 decode 極大化（2026-11-27）

> 配置：Q4_K_M + MTP draft Q4_0，ctx4096。
> 環境：swap=0、bg=1、power 7.7–8.2W、temp 62°C。

## 一、thinking ON 長輸出（decode 極致的關鍵場景）

| n-max | decode | 接受率 |
|---|---:|---:|
| n-max4 | 40.0 t/s | 60% |
| **n-max6** | **40.7 t/s** | 48% |
| n-max8 | 41.1 t/s | 42% |

→ **n-max 對思考內容影響 <3%**（4/6/8 都 ~40–41）。思考內容已接近頻寬極限。

## 二、thinking ON vs OFF（128 tok）

| 模式 | decode | E2E | 接受率 |
|---|---:|---:|---:|
| **thinking ON** | **40.8 t/s** | 3387 ms | 48% |
| thinking OFF | 26.5 t/s | 5086 ms | 22% |

→ thinking ON 讓 decode 從 26.5 → 40.8（+54%），E2E 反而更短（思考內容可預測、接受率高）。

## 三、raw /completion 長文（任意內容）

| n-max | decode | 接受率 |
|---|---:|---:|
| n-max4 | 32.3 t/s | 39% |
| n-max6 | 29.4 t/s | 25% |

→ 任意長文 decode 26–32 t/s（內容不可預測，接受率低）。

## 四、結論

| 內容型態 | decode | 說明 |
|---|---:|---|
| **思考/推理** | **40.8** | 🏆 已達標（目標 50 差 18%）|
| 一般長文 | 26–32 | 受接受率限制 |
| 任意 ignore_eos | 26 | 最低 |

**50 t/s 的瓶頸**：思考內容 40.8，理論值（L_eff 3.82 × BW 62 / 4.64）= 51。**gap 21% = verify/同步開銷** → 需 Phase 5b 改 kernel（2-row verify / Selected 16K）。

## 五、與歸檔對照
- 歸檔 40.5（n-max6/t2）→ 本次 40.7 ✅ 重現。
- 歸檔「任意長文 18.2」→ 本次 26（用 nav prompt 較可預測）。

## 檔案
- `raw/P1_B1_*`（raw 長文）
- thinking 測試（inline，見 runsheet 未含，補記於本報告）