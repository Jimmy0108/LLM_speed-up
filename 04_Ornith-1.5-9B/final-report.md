# 最終端到端比較報告

> 日期：2026-11-25  
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER  
> JetPack 7.2.2 / CUDA 13.2

> ⚠️ **數據修正（2026-11-26）**：下表 llama.cpp prefill「**4829 t/s**」是重複 prompt 造成的 **prefix cache 汙染**，**不可用**。
> 同條件公平實測（unique prompt）請看 `05_全模型消融實驗_20261126`：Gemma4 prefill ~248 t/s、Ornith ~158–171 t/s。
> 本表其餘 TTFT / decode / RAM 仍可參考。

---

## 一、Gemma 4 E4B（4.10B）Edge-LLM INT4 vs llama.cpp Q4_0

| 指標 | Edge-LLM INT4 | llama.cpp Q4_0 | 贏家 |
|---|---|---|---|
| TTFT median | 142ms | 138ms | 平手 |
| Decode | **17.0 t/s** | 16.3 t/s | Edge-LLM +4% |
| Prefill (512 tok) | 796 t/s | **4829 t/s** | llama.cpp **6x** |
| Sustained (20 req) | **1189ms** | 2141ms | Edge-LLM +44% |
| RAM | 15.0 GB | **8.5 GB** | llama.cpp **省 6.5GB** |
| 剩餘可用 | 0.2 GB ⚠️ | 3.3 GB ✅ | llama.cpp |

## 二、Qwen3.5-9B（9B）Edge-LLM INT4

| 指標 | Edge-LLM INT4 |
|---|---|
| TTFT median | 181ms |
| Decode | 10.0 t/s |
| Prefill (512 tok) | 509 t/s |
| RAM | 14.6 GB（剩 0.3 GB ⚠️） |

## 三、Ornith-1.5-9B Edge-LLM 可行性

| 嘗試 | 結果 | 原因 |
|---|---|---|
| FP16 直接跑 | ❌ | 18GB > 11GB 可用 |
| 本地 AWQ 量化 | ❌ | FP16 forward pass 需要 18GB，16GB Orin OOM |
| cyankiwi AWQ (gs=32) | ❌ build OK, runtime 失敗 | v2 plugin 只支援 gs=128；v1 plugin SM87 執行失敗 |
| Sohailhosseini AWQ (gs=128, sym=False) | ❌ build 失敗 | asymmetric 需要 zero_point tensor，未提供 |
| Sohailhosseini AWQ runtime | ❌ runtime 失敗 | `linear_attn.in_proj_qkv.qweight` binding 不相容 |
| **結論** | **現階段無法在 Edge-LLM 上跑** | 無相容的 gs=128 + symmetric=True checkpoint |

### Ornith 替代方案

| 方案 | 可行性 | 說明 |
|---|---|---|
| A. x86 GPU 做 AWQ 量化 | ✅ 可行 | 在 x86 上跑 `trtedge quantize`，搬 INT4 到 Orin build |
| B. 用 Qwen3.5-9B INT4 代替 | ✅ 已驗證 | 同架構、非 Ornith 微調、10.0 t/s |
| C. 維持 llama.cpp Q4_K_M | ✅ 已驗證 | 12.1 t/s、省 5GB RAM、24/7 穩定 |

## 四、綜合建議

| 場景 | 建議 | 理由 |
|---|---|---|
| 4B 模型、效能優先 | Edge-LLM INT4 | decode 17 t/s、TTFT 142ms |
| 4B 模型、穩定優先 | **llama.cpp Q4_0** | 省 6.5GB、prefill 6x 快 |
| 9B 模型 | **llama.cpp Q4_K_M** | 12.1 t/s、穩定、Edge-LLM 記憶體不足 |
| Ornith-1.5-9B | **llama.cpp Q4_K_M** | Edge-LLM 無相容 INT4 checkpoint |
| 未來 Ornith on Edge-LLM | x86 量化 → Orin build | 需要 group_size=128 + symmetric=True |