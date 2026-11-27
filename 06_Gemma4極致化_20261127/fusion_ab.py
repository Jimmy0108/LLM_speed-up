#!/usr/bin/env python3
"""Fusion / CUDA-graph A/B：固定模型與 flag，只改 env。
用法: fusion_ab.py --tag base
"""
import argparse, json, random, statistics, time, urllib.request, os

CHT = "http://127.0.0.1:8080/v1/chat/completions"
JSON_P = "用 JSON 陣列列出三個常見水果，每個含 name 與 color 欄位。"
PROSE_P = "詳細解釋光合作用的完整過程。"
LONG_P = ("請詳細說明下列主題，並分成多個段落：Jetson Orin NX 的統一記憶體架構、"
          "LPDDR5 頻寬對 LLM decode 的影響、MTP 投機解碼的接受率、"
          "CUDA Graph 與 kernel launch 開銷、算子融合如何減少 DRAM 讀寫、"
          "以及 prefill 與 decode 的瓶頸差異。" * 4)


def chat(prompt, max_tokens, reps, label, out):
    rows = []
    for i in range(reps):
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            CHT, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        wall = (time.perf_counter() - t0) * 1000
        t = r.get("timings", {})
        g = t.get("draft_n", 0)
        a = t.get("draft_n_accepted", 0)
        row = {
            "label": label,
            "rep": i,
            "predicted_n": t.get("predicted_n", 0),
            "decode_tps": round(t.get("predicted_per_second", 0), 2),
            "accept": round(a / g, 4) if g else 0.0,
            "prompt_n": t.get("prompt_n", 0),
            "prompt_tps": round(t.get("prompt_per_second", 0), 2),
            "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "e2e_ms": round(wall, 1),
        }
        rows.append(row)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
    dec = [x["decode_tps"] for x in rows]
    acc = [x["accept"] for x in rows if x["accept"]]
    e2e = [x["e2e_ms"] for x in rows]
    ptps = [x["prompt_tps"] for x in rows if x["prompt_tps"]]
    s = {
        "label": label,
        "decode_median": round(statistics.median(dec), 1),
        "decode_mean": round(statistics.mean(dec), 1),
        "accept": round(statistics.mean(acc), 3) if acc else 0.0,
        "e2e_median": round(statistics.median(e2e), 0),
        "prompt_tps_median": round(statistics.median(ptps), 1) if ptps else 0.0,
    }
    print(json.dumps(s, ensure_ascii=False), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--prefill-reps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs("raw", exist_ok=True)
    out_path = f"raw/fusion_{args.tag}.jsonl"
    sums = []
    with open(out_path, "w", encoding="utf-8") as out:
        sums.append(chat(JSON_P, 128, args.reps, f"{args.tag}_json", out))
        sums.append(chat(PROSE_P, 128, args.reps, f"{args.tag}_prose", out))
        sums.append(chat(LONG_P, 1, args.prefill_reps, f"{args.tag}_prefill", out))
    print("SUMMARY_JSON " + json.dumps(sums, ensure_ascii=False))


if __name__ == "__main__":
    main()
