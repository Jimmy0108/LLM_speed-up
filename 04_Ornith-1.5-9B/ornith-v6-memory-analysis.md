# Ornith-1.5-9B v6 × llama.cpp：記憶體佔用分析

> 量測時間：2026-11-25
> 服務：`ornith-v6.service`（MTP n=3、`-c 2048`、`-ngl 99`、`-fa on`）
> 單位：MiB / GiB（1 GiB = 1024 MiB）；括號內為十進位 GB

---

## 1. GGUF 檔案本身

| 項目 | 大小 |
|---|---:|
| 檔案實際 bytes | 5,831,992,096 |
| 檔案大小 | **5.43 GiB（5.83 GB）** |
| 其中 tensor 權重 | 5.42 GiB（5.82 GB）|
| metadata / 對齊 | 10.5 MiB |

量化組成：

| 類型 | 大小 |
|---|---:|
| Q4_K（type 12） | 3.51 GiB |
| Q6_K（type 14） | 1.91 GiB |
| F32（norm 等） | 0.004 GiB |

> 這是 **v6**（含 LoRA）的 GGUF；`~/local-llm/models/Ornith-1.5-9B-Q4_K_M.gguf`（5.78 GB）是舊的 base，不是這次用的。

---

## 2. llama.cpp 自己回報的用量

啟動時 `-lv 4` 的實測值：

| 元件 | MiB | GiB |
|---|---:|---:|
| CUDA0 模型權重（GPU） | 5,005.74 | 4.89 |
| CUDA_Host 模型權重（Host） | 545.62 | 0.53 |
| KV cache（target，2048 ctx） | 64.00 | 0.06 |
| GDN recurrent state | 201.00 | 0.20 |
| target compute buffer（CUDA + Host） | 57.10 + 17.01 | 0.07 |
| MTP KV cache | 8.00 | 0.01 |
| MTP compute buffer（CUDA + Host） | 45.01 + 9.01 | 0.05 |
| MTP h_nextn buffer | 8.02 | 0.01 |
| Host output buffer | 0.95 | 0.001 |
| **llama.cpp 合計** | **≈ 5,960** | **≈ 5.82** |

原始 log：

```
load_tensors: offloaded 34/34 layers to GPU
load_tensors:        CUDA0 model buffer size =  5005.74 MiB
load_tensors:    CUDA_Host model buffer size =   545.62 MiB
llama_kv_cache:      CUDA0 KV buffer size =    64.00 MiB
llama_memory_recurrent: CUDA0 RS buffer size = 201.00 MiB
sched_reserve:       CUDA0 compute buffer size = 57.10 MiB
sched_reserve:   CUDA_Host compute buffer size = 17.01 MiB
MTP: KV 8.00 MiB, compute 45.01 + 9.01 MiB, h_nextn 8.02 MiB
```

llama.cpp 的 device memory breakdown：

```
CUDA0 (Orin) 15533 = 4034 + (5327 = 5005 + 265 + 57) + 6170
Host          562 =  545 +    0 +  17
```

- `5327 MiB` = llama.cpp 自己在 GPU 上的 model + context + compute
- `6170 MiB` 是「unaccounted」，包含其他 GPU 使用者（Xorg、Chrome GPU）與 CUDA driver/runtime，不是 llama-server 獨占

---

## 3. 作業系統看到的 llama-server

| 指標 | 值 |
|---|---:|
| VmRSS（process RSS） | **6.35 GiB（6.82 GB）** |
| 其中 RssAnon | 5.76 GiB |
| 其中 RssFile | 0.59 GiB |
| VmHWM（peak RSS） | 6.35 GiB |
| VmSwap | 0（完全沒被換出）|
| cgroup `memory.current` | **6.78 GiB（7.28 GB）** |
| cgroup `memory.peak` | **6.94 GiB（7.45 GB）** |
| 先前完整 benchmark 的 cgroup peak | 7.24 GiB（7.78 GB）|

cgroup 分類：

| 類別 | GiB | 說明 |
|---|---:|---|
| `kernel` | 4.24 | Jetson GPU/NvMap 的權重配置（主要成本）|
| `file` | 2.13 | GGUF 的 page cache，**可回收** |
| `anon` | 0.41 | heap / compute buffer 等 |
| `pagetables` / `slab` | <0.01 | |

> Jetson 是 unified memory，GPU 配置在 cgroup 會算進 `kernel`，在 `/proc` 又會反映在 `RssAnon`；
> 兩個 view 的數字不會完全相加，這是 driver 記帳方式造成的，不是漏算。

### 為什麼 RSS（6.35）比 llama.cpp 帳面（5.82）多？

差額約 **0.53 GiB**，主要是：

- CUDA runtime / driver context
- process heap（約 197 MiB）
- 動態 library（libcublasLt、libcuda、libggml-cuda…）
- dmabuf / NvMap 映射（約 546 MiB，對應 Host model buffer）

---

## 4. 整體系統

| 項目 | MiB | GiB |
|---|---:|---:|
| 總 RAM | 15,533 | **15.17** |
| 已使用 | 11,658 | **11.39** |
| 可用（available） | 3,875 | **3.78** |
| buff/cache | 3,432 | 3.35 |
| Swap 已使用 | 1,411 | 1.38 |

### 主要佔用者

| 程式 | RSS |
|---|---:|
| **llama-server（Ornith v6 + MTP）** | **6.35 GiB（約 used 的 56%）** |
| Chrome（所有 process 合計） | 2.35 GiB |
| GNOME Shell | 0.40 GiB |
| opencode（serve + client） | 0.58 GiB |
| Xorg | 0.25 GiB |
| 其他系統服務 / driver | 約 1.4 GiB |

### Swap 使用者

Swap 1.38 GiB 全部是桌面程式，llama-server 的 `VmSwap = 0`：

- GNOME Shell 108 MiB
- update-manager 103 MiB
- xdg-desktop-portal-gnome 89 MiB
- Nautilus 73 MiB
- Chrome GPU process 66 MiB
- gnome-software 61 MiB

---

## 5. 結論

1. **GGUF 5.83 GB**，載入後權重約 **5.42 GiB**。
2. **llama.cpp 帳面用量 5.82 GiB**：
   - 權重 5.42 GiB（GPU 4.89 + Host 0.53）
   - KV + GDN state 0.27 GiB
   - compute buffer 0.12 GiB
   - MTP 額外 0.07 GiB
3. **llama-server 實際佔用**：
   - RSS **6.35 GiB**
   - cgroup **6.78 GiB**（peak 6.94 GiB，benchmark 曾到 7.24 GiB）
   - 其中 2.13 GiB 是可回收的 page cache
4. **系統**：總 15.17 GiB、已用 11.39 GiB、可用 3.78 GiB、swap 1.38 GiB。
5. **MTP 很划算**：只多約 70 MiB 帳面記憶體，decode 從 10.26 → 18.65 tok/s（1.8x）。
6. **2048 ctx 的 KV/state 很小**（64 + 201 MiB）；就算開到 4096，也只多約 70 MiB。
7. 目前 `MemoryMax=12G`，距離上限還有約 5 GiB；真正的壓力來源是 Chrome/桌面，不是 llama-server。
