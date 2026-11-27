# G 系列 runsheet（機器/人類共用）

所有數字為中位數；逐筆見同名 jsonl。

## 1. Fusion / Graph / Precision（decode t/s、prefill t/s）

| tag | 設定 | JSON decode | 散文 decode | prefill | JSON accept |
|---|---|---:|---:|---:|---:|
| base | 預設（fusion ON, graphs ON） | 66.71 | 25.15 | 398.8 | 0.75 |
| mmqq4 | GGML_CUDA_MMQ_PREC=q4 | 65.53 | 24.32 | 389.1 | 0.75 |
| graphopt | GGML_CUDA_GRAPH_OPT=1 | 66.05 | 25.04 | 395.9 | 0.75 |
| nofuse | GGML_CUDA_DISABLE_FUSION=1 | 66.02 | 23.4 | 384.3 | 0.75 |
| mmqq8 | GGML_CUDA_MMQ_PREC=q8 | 66.11 | 24.57 | 392.3 | 0.75 |
| nograph | GGML_CUDA_DISABLE_GRAPHS=1 | 59.45 | 22.1 | 390.3 | 0.75 |

## 2. 短答（nav_testset, lite rules）

| tag | 設定 | n | EM | F1 | E2E 中位 | prefill 中位 | 100 次 p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| ub256 | -np 1, -b1024 -ub256（30 題） | 30 | 0.9667 | 0.9889 | 495 | 251 | 515 |
| ub1024 | -np 1, -b2048 -ub1024（30 題） | 30 | 0.9667 | 0.9889 | 628 | 330 | 685 |
| cache64 | -np 1 + cache_prompt=true + --cache-reuse 64（無效） | 124 | 0.8952 | 0.9742 | 594 | 337 | 509 |
| ub512 | -np 1, -b1024 -ub512（30 題） | 30 | 0.9667 | 0.9889 | 493 | 248 | 680 |
| np4 | -np 4（30 題） | 30 | 0.9667 | 0.9889 | 600 | 357 | 619 |
| nocache | -np 1, cache_prompt=false（正式） | 124 | 0.8952 | 0.9742 | 536 | 326 | 520 |
| np1 | -np 1（30 題） | 30 | 0.9667 | 0.9889 | 498 | 248 | 516 |

## 3. 硬體 / Roofline

| 項目 | 值 |
|---|---:|
| 模型檔總大小 | 4.961 GB |
| PLE 表（按列取） | 1.938 GB |
| tied lm_head / token_embd | 0.377 GB |
| transformer 權重 | 2.591 GB |
| **每步實際串流** | **3.023 GB** |
| 實測 read-only | 57.3 GB/s |
| 實測 D2D copy | 66.4 GB/s |
| 自由散文 decode 頻寬 | 48.4 GB/s（屋頂 73–84%） |
| JSON n8 decode 頻寬 | 28.8 GB/s（屋頂 44–50%） |

## 4. 環境

- 未鎖頻（nvhost_podgov），EMC 3199 MHz，GR3D 91–92%，功耗 20.4–20.7W，≤73°C
- ncu 不可用（統一記憶體配置失敗 error 12）；以 nsys + 自寫 bandwidth 測試替代
- nsys kernel 原始表：`raw/nsys_kern_sum.txt`
