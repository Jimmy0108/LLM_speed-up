# C1 — Adaptive Draft Length 最終報告（2026-11-27）

> 模型：Gemma 4 E4B Q4_K_M + MTP draft Q4_0（不換、不下載）
> 改動：llama.cpp `common/speculative.cpp` MTP class + `common.h` + `arg.cpp`（新增 `--spec-adaptive`）
> 硬體：Orin NX 16GB / MAXN_SUPER / SM87 / FA=ON

## 一、實作
- 新增 CLI：`--spec-adaptive` / `--spec-adapt-lo P` / `--spec-adapt-hi P`。
- MTP class 加 rolling 接受率 EWMA（α=0.3），每輪 draft 前依門檻 ±1 調 `n_max`，界 `[floor, cap]`。
- 投機 lossless → 只改 n_max，不改輸出。

## 二、正確性（閘門通過）
固定 n-max6 與 adaptive 的 greedy 輸出 **byte-identical**（3 題 × 24 token 完全相同）→ lossless 確認。

## 三、思考內容 decode（128 tok, thinking ON）

| 配置 | decode | 接受率 |
|---|---:|---:|
| **fixed n-max6** | **42.4 t/s** 🏆 | 48% |
| fixed n-max4 | 40.5 | 60% |
| adapt cap6 | 41.1 | 50% |
| fixed n-max8 | 39.2 | 41% |
| adapt cap4 | 38.6 | 59% |
| fixed n-max10 | 38.0 | 35% |
| KV q4_0 | 38.7 | 44% |
| fixed n-max12 | 36.4 | 32% |
| FA off | ❌ 壞（sm_87 bug） | — |

→ **n-max6 固定是思考峰值 42.4**；adaptive 打不贏（38–41）；n>6 變慢；KV q4_0 變慢；FA off 不可用。

## 四、短答 E2E（lite prompt）

| 配置 | E2E | decode |
|---|---:|---:|
| fixed n-max3 | 693 ms | 32.3 |
| **adapt cap3** | 708 ms | **36.2** |
| fixed n-max6 | 727 ms | 29.4 |
| adapt cap6 | 835 ms | 24.6 |

→ adaptive cap3 的 decode 較高（36.2 vs 32.3），但 E2E 在噪音內（n=3）；cap6 明顯變差。

## 五、結論（誠實）
1. **C1 adaptive 是 lossless 的「不確定最佳固定值時」保險**，不是峰值加速。
2. 我們的 E4B 已手調到最佳固定值（思考 n-max6、短答 n-max3），**adaptive 無法再贏**（符合 BanditSpec/SpecDec++ 預估：已選對固定值時增益 ~0–3%）。
3. **思考 decode 結構天花板 = 42.4 t/s**（n-max6）。50 t/s 在 Q4_K_M 上不可達：
   - GPU 97% memory-bound、mmq-few 已 optimal、n>6 verify 成本吃掉收益。
   - 要 50+ 只能 QAT（排除）或改硬體。
4. **2-row verify 不移植**：我們 build 已含 mmq-few（B=4 從 1.49×→1.05×），B=2 僅 ~3% 窄空間。

## 六、保留的改動
- `--spec-adaptive` 已 merge 進本地 build（可選用，不影響預設行為）。
- 對「未知/混合工作負載」有 +5–15% 潛力（短答避免固定 n 的懲罰）。
- 對「已知最佳固定值」無增益。

## 七、數據
- `raw/C1_summary.json` — 全部 A/B 數據
- `c1_ab.py` — A/B harness
- `server-C1*.log` — server 原始 log