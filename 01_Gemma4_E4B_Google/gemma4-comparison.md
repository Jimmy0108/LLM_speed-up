# Gemma 4 E4B 三方比較報告

> 日期：2026-11-25  
> 硬體：Jetson Orin NX 16GB / MAXN_SUPER  
> JetPack 7.2.2 / CUDA 13.2

---

## 測試配置

| 框架 | 精度 | 量化格式 | 模型大小 |
|---|---|---|---|
| Edge-LLM INT4 | W4A16 | compressed-tensors (gs=128) | 8.7 GB |
| llama.cpp | Q4_0 | GGUF QAT | 4.9 GB |

模型：google/gemma-4-E4B-it（42 層, hidden=2560, 8Q/2KV heads, 256K context）

---

## 效能比較

| 指標 | Edge-LLM INT4 | llama.cpp Q4_0 | 贏家 |
|---|---|---|---|
| **TTFT** | **128ms** 🏆 | 125ms | 平手 |
| **Prefill (512 tok)** | **885 t/s** 🏆 | 555 t/s | Edge-LLM +59% |
| **Decode** | **18.6 t/s** 🏆 | 18.0 t/s | Edge-LLM +3% |
| RAM | 14.0 GB | 8.2 GB | llama.cpp 🏆 |
| 剩餘可用 | 0.7 GB | 6.9 GB | llama.cpp 🏆 |

---

## 與其他模型比較

| 模型 | 框架 | TTFT | Prefill | Decode | RAM |
|---|---|---|---|---|---|
| **Gemma 4 E4B** | **Edge-LLM INT4** | **128ms** | **885 t/s** | **18.6 t/s** | 14.0 GB |
| Gemma 4 E4B | llama.cpp Q4_0 | 125ms | 555 t/s | 18.0 t/s | 8.2 GB |
| Qwen3.5-4B | Edge-LLM INT4 | 127ms | 808 t/s | 20.8 t/s | 6.9 GB |
| Qwen3.5-9B | Edge-LLM INT4 | 157ms | 605 t/s | 11.5 t/s | 13.0 GB |
| Ornith-1.5-9B | llama.cpp Q4_K_M | ~400ms | 360 t/s | 12.1 t/s | 10.0 GB |

---

## 結論

### Gemma 4 E4B 是目前最佳選擇

1. **非中國模型**：Google 美國公司，Apache 2.0
2. **效能強**：decode 18.6 t/s，prefill 885 t/s（比 Ornith 9B 快 2.5x）
3. **Edge-LLM 完整支援**：group_size=128 相容，無需 patch
4. **LoRA 微調**：效果不好可以自己調整

### 記憶體問題

Edge-LLM INT4 用了 14GB（剩 0.7GB），不適合 24/7 長時間運行。
llama.cpp Q4_0 只用 8.2GB（剩 6.9GB），更穩定。

### 建議

- **效能優先**：Edge-LLM INT4（TTFT 128ms, prefill 885 t/s）
- **穩定優先**：llama.cpp Q4_0（省 6GB RAM, decode 相近）
- **4B 最佳平衡**：Qwen3.5-4B Edge-LLM INT4（20.8 t/s decode, 6.9GB RAM）