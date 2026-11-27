#!/usr/bin/env python3
"""格式頻譜 benchmark：固定引擎參數，只改輸出格式，量 MTP 接受率與 decode。
用法: format_spectrum.py --tag j8 --nmax 8 --suite full
      format_spectrum.py --tag p4 --nmax 4 --suite prose
"""
import argparse, json, random, statistics, time, urllib.request, os

CHT = "http://127.0.0.1:8080/v1/chat/completions"

PROSE_OPEN = [
    "詳細解釋光合作用的完整過程。",
    "為什麼天空是藍色？請詳細分析。",
    "證明勾股定理，並說明它的意義。",
    "解釋量子糾纏的原理與實際應用。",
    "描述法國大革命的三個主要階段與影響。",
]
PROSE_STEPS = [
    "詳細解釋光合作用全過程，分步骤說明。",
    "為什麼天空是藍的？詳細分析瑞利散射。",
    "證明勾股定理，並用兩種不同方法說明。",
    "解釋量子糾纏的原理與應用，分步骤說明。",
    "描述法國大革命的三個主要階段。",
]
BULLETS = [
    "用列點列出 5 個提升 GPU 推理速度的方法，每點一句說明。",
    "用列點列出 5 個降低 LLM 記憶體佔用的技巧，每點一句說明。",
    "用列點列出 5 個 Jetson 省電的方法，每點一句說明。",
    "用列點列出 5 個加速 prefill 的方向，每點一句說明。",
    "用列點列出 5 個提升 MTP 接受率的做法，每點一句說明。",
]
TABLE = [
    "用 Markdown 表格列出 6 個國家，欄位：國家、首都、人口。",
    "用 Markdown 表格列出 6 個程式語言，欄位：名稱、年份、作者。",
    "用 Markdown 表格列出 6 個行星，欄位：名稱、順序、直徑。",
    "用 Markdown 表格列出 6 個資料結構，欄位：名稱、查詢複雜度、插入複雜度。",
    "用 Markdown 表格列出 6 個量化格式，欄位：名稱、bits、典型用途。",
]
JSONQ = [
    "用 JSON 陣列列出 6 個常見水果，每個含 name、color、price 三個欄位。",
    "用 JSON 陣列列出 6 個程式語言，每個含 name、year、author 三個欄位。",
    "用 JSON 陣列列出 6 個行星，每個含 name、order、diameter 三個欄位。",
    "用 JSON 陣列列出 6 個資料結構，每個含 name、query、insert 三個欄位。",
    "用 JSON 陣列列出 6 個量化格式，每個含 name、bits、use 三個欄位。",
]
CODE = [
    "寫一個 Python 函式計算 Fibonacci 第 n 項，含 docstring 與範例。",
    "寫一個 Python 函式判斷質數，含 docstring 與範例。",
    "寫一個 Python 函式做二分搜尋，含 docstring 與範例。",
    "寫一個 Python 函式反轉字串，含 docstring 與範例。",
    "寫一個 Python 函式計算最大公因數，含 docstring 與範例。",
]
JSON_SAME = ["用 JSON 陣列列出三個常見水果，每個含 name 與 color 欄位。"] * 10

SUITES = {
    "prose": [("prose_open", PROSE_OPEN), ("prose_steps", PROSE_STEPS)],
    "full": [
        ("prose_open", PROSE_OPEN),
        ("prose_steps", PROSE_STEPS),
        ("bullets", BULLETS),
        ("table", TABLE),
        ("json", JSONQ),
        ("code", CODE),
        ("json_same", JSON_SAME),
    ],
}


def chat(prompt, max_tokens):
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
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    wall = (time.perf_counter() - t0) * 1000
    return resp, wall


def run_cat(tag, nmax, cat, prompts, reps, max_tokens, out_fh):
    rows = []
    for p in prompts:
        for k in range(reps):
            resp, wall = chat(p + f" [u-{random.randint(0, 10**9)}]", max_tokens)
            t = resp.get("timings", {})
            g = t.get("draft_n", 0)
            a = t.get("draft_n_accepted", 0)
            row = {
                "tag": tag,
                "nmax": nmax,
                "category": cat,
                "max_tokens": max_tokens,
                "rep": k,
                "predicted_n": t.get("predicted_n", 0),
                "predicted_ms": round(t.get("predicted_ms", 0), 1),
                "decode_tps": round(t.get("predicted_per_second", 0), 2),
                "draft_n": g,
                "draft_accepted": a,
                "accept": round(a / g, 4) if g else 0.0,
                "e2e_ms": round(wall, 1),
                "content_chars": len(resp["choices"][0]["message"]["content"]),
                "finish": resp["choices"][0].get("finish_reason"),
            }
            rows.append(row)
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()
    dec = [r["decode_tps"] for r in rows]
    acc = [r["accept"] for r in rows if r["draft_n"]]
    e2e = [r["e2e_ms"] for r in rows]
    pn = [r["predicted_n"] for r in rows]
    summary = {
        "category": cat,
        "n": len(rows),
        "decode_median": round(statistics.median(dec), 1),
        "decode_mean": round(statistics.mean(dec), 1),
        "decode_min": round(min(dec), 1),
        "decode_max": round(max(dec), 1),
        "accept_mean": round(statistics.mean(acc), 3) if acc else 0.0,
        "e2e_median": round(statistics.median(e2e), 0),
        "pred_n_median": round(statistics.median(pn), 0),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nmax", type=int, required=True)
    ap.add_argument("--suite", default="prose", choices=["prose", "full"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--length-test", action="store_true")
    args = ap.parse_args()

    os.makedirs("raw", exist_ok=True)
    out_path = f"raw/spectrum_{args.tag}.jsonl"
    summaries = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for cat, prompts in SUITES[args.suite]:
            summaries.append(
                run_cat(args.tag, args.nmax, cat, prompts, args.reps, args.max_tokens, fh)
            )
        if args.length_test:
            for cat, prompts in [("prose_open_512", PROSE_OPEN), ("json_512", JSONQ)]:
                summaries.append(
                    run_cat(args.tag, args.nmax, cat, prompts[:3], 3, 512, fh)
                )
    print("SUMMARY_JSON " + json.dumps({"tag": args.tag, "nmax": args.nmax, "cats": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
