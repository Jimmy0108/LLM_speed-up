# Phase 3/4 — 雙路徑 + 功耗（2026-11-27）

## Phase 3 雙路徑驗證

單一 server（ctx4096 n-max6 b1024 t4），短答用 lite prompt、長答用 thinking：

| 路徑 | E2E | decode |
|---|---:|---:|
| 短答（lite prompt 62 tok） | 746 ms | 24.1 |
| 長答（thinking 128 tok） | 3176 ms | **44.4** |

→ 同一 server 可同時服務兩種；短答靠「短 prompt」降 prefill，長答靠「thinking」提接受率。

## Phase 4 功耗（3 分鐘滿載）

| 指標 | 值 |
|---|---|
| 功耗（VDD_IN 整板） | 中位 **20.6 W**、峰值 21.5 W |
| GPU 利用率 | 中位 97%、峰值 99% |
| GPU 溫度 | 中位 70.4°C、峰值 72.7°C |
| 降頻 | 無（遠低於 ~95°C 限） |
| swap | 0（全程） |

→ 滿載穩定、無 thermal throttle、無 swap I/O 汙染。