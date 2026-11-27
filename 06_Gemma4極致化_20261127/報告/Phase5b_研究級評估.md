# Phase 5b — 研究級改碼可行性評估（2026-11-27）

> 目標：把思考內容 decode 從 40.8 推向 50+。
> 瓶頸：40.8 實測 vs 51 理論（L_eff 3.82 × BW 62 / 4.64）→ gap 21% = verify/同步開銷。

## 一、上游已具備的基礎（好消息）

| 基礎 | 檔案 | 狀態 |
|---|---|---|
| **Few-row MMQ**（2–24 列小矩陣） | `ggml/src/ggml-cuda/mmq-few.cu` | ✅ 上游已有，正是 MTP verify（B=2–24）情境 |
| MTP 投機框架 | `common/speculative.cpp`（3295 行） | ✅ 完整，含 draft/accept 統計 |
| CUDA Graph | `GGML_CUDA_GRAPHS=ON` | ✅ 已編譯 |
| p-min early-stop | `--spec-draft-p-min` | ✅ 已用 |

→ **little-gemma 的「2-row verify」是 mmq-few 的再特化**（一 warp 負責兩輸出列）。我們已有 mmq-few，5b 是「調校 + 確認是否吃滿」而非從零寫。

## 二、三個改碼項評估

| 項 | 改動 | 預估 | 風險 | 建議 |
|---|---|---|---|---|
| **C1 Adaptive draft**（依 rolling r 調 n） | `speculative.cpp` 加控制器 | +5–15% | 中 | ✅ 值得（CP 高）|
| **C2 2-row verify 調校** | `mmq-few.cu` 調 block/warp | +10–20% | 高 | ⚠️ 先量 mmq-few 是否已吃滿 |
| **C3 Selected 16K Vocab** | draft 詞彙裁剪 | +5–8% | 中 | ⚠️ 需 ID list，little-gemma 專有 |

## 三、建議執行順序（5b）
1. **先量 mmq-few 使用率**：用 nsys 看 MTP verify 是否走 mmq-few、GPU 利用率。若已 97%（Phase 4 實測），改 kernel 空間有限。
2. **C1 Adaptive draft**：改 `speculative.cpp` 依 rolling accept 調 n-max（低 r 降 n、高 r 升 n）。CP 最高、風險最低。
3. C2/C3 若 nsys 顯示 verify 未成瓶頸 → 標記「不需改」。

## 四、誠實結論
- Phase 4 實測 GPU 利用率 **97%**，代表**硬體已近吃滿**，單純改 kernel 的 50+ 空間比想像小。
- 40.8 → 50 的 21% gap，主因是 **MTP verify 的序列依賴**（每輪 draft→verify 無法完全 parallel）。
- **C1 Adaptive draft 是最務實的 5b**（+5–15%，把 40.8 推到 ~43–47）。
- 要真正 50+ 可能需 C1+C2 合力，且 C2 風險高。

## 五、下一步
- 待使用者決定是否投入 C1 改碼（需重編 llama.cpp，~10–20 分鐘 build）。
- 或接受 40.8（思考）/ 44.4（雙路徑長答）為「不改碼極限」。