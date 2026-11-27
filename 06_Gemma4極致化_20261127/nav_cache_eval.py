#!/usr/bin/env python3
"""G08 + G13：prompt prefix cache 對短答的 E2E / 正確率 / 100 次穩定性。

用法（server 需開 --cache-reuse 64 -np 1）：
    nav_cache_eval.py --tag cache64
輸出：raw/nav_cache_<tag>.jsonl、raw/nav_cache_<tag>_summary.json
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, "/home/jetson/Desktop/for_jetson")
import eval_jetson as ev  # noqa: E402

NAV = "/home/jetson/Desktop/for_jetson/datasets/nav_testset.jsonl"


def complete(url, prompt, max_tokens=64, cache=True):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "stop": ["\n"],
        "cache_prompt": cache,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return (time.perf_counter() - t0) * 1000, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080/completion")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    use_cache = not args.no_cache
    os.makedirs("raw", exist_ok=True)
    out = open(f"raw/nav_cache_{args.tag}.jsonl", "w", encoding="utf-8")

    rows = [json.loads(l) for l in open(NAV, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    complete(args.url, "暖機", 4, cache=use_cache)

    items = []
    for i, r in enumerate(rows, 1):
        p = ev.build_raw_prompt(r["prompt"], rules_mode="lite")
        e2e, resp = complete(args.url, p, 64, cache=use_cache)
        raw = (resp.get("content") or "").strip()
        t = resp.get("timings", {})
        pred = ev.parse_tokens(raw)
        gold = [str(g).strip().lower() for g in r["expected"]]
        _, _, f1 = ev.set_scores(pred, gold)
        rec = {
            "kind": "nav", "id": r["id"], "cat": r.get("category", "?"),
            "em": pred == gold, "f1": round(f1, 4), "raw": raw,
            "e2e_ms": round(e2e, 1),
            "prompt_n": t.get("prompt_n", 0),
            "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "predicted_n": t.get("predicted_n", 0),
            "decode_tps": round(t.get("predicted_per_second", 0), 2),
            "draft_n": t.get("draft_n", 0),
            "draft_accepted": t.get("draft_n_accepted", 0),
        }
        items.append(rec)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if i % 20 == 0:
            print(f"  nav {i}/{len(rows)}", flush=True)

    n = len(items)
    em = sum(x["em"] for x in items) / n
    f1 = statistics.mean(x["f1"] for x in items)
    e2e = [x["e2e_ms"] for x in items]
    warm = e2e[1:]
    pm = [x["prompt_ms"] for x in items[1:]]
    print(f"NAV n={n} EM={em:.4f} F1={f1:.4f} "
          f"E2E中位={statistics.median(e2e):.0f}ms 暖機後中位={statistics.median(warm):.0f}ms "
          f"prompt_ms中位={statistics.median(pm):.0f}", flush=True)

    # 100 次連續請求穩定性（固定短答 prompt）
    stab_prompt = ev.build_raw_prompt("從大廳往會議室走，途中經過櫃檯", rules_mode="lite")
    stab = []
    t0 = time.perf_counter()
    for i in range(args.reps):
        e2e_ms, resp = complete(args.url, stab_prompt, 32, cache=use_cache)
        t = resp.get("timings", {})
        rec = {
            "kind": "stability", "rep": i, "e2e_ms": round(e2e_ms, 1),
            "prompt_n": t.get("prompt_n", 0),
            "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "predicted_n": t.get("predicted_n", 0),
            "decode_tps": round(t.get("predicted_per_second", 0), 2),
            "draft_n": t.get("draft_n", 0),
            "draft_accepted": t.get("draft_n_accepted", 0),
            "content": (resp.get("content") or "").strip(),
        }
        stab.append(rec)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if (i + 1) % 20 == 0:
            print(f"  stab {i+1}/{args.reps}  {time.perf_counter()-t0:.0f}s", flush=True)
    out.close()

    se = [x["e2e_ms"] for x in stab]
    sd = [x["decode_tps"] for x in stab]
    spm = [x["prompt_ms"] for x in stab]
    uniq = len({x["content"] for x in stab})
    summary = {
        "tag": args.tag, "nav_n": n, "em": round(em, 4), "f1": round(f1, 4),
        "nav_e2e_median_ms": round(statistics.median(e2e), 1),
        "nav_e2e_warm_median_ms": round(statistics.median(warm), 1),
        "nav_prompt_ms_median": round(statistics.median(pm), 1),
        "stab_n": len(stab),
        "stab_e2e_median_ms": round(statistics.median(se), 1),
        "stab_e2e_p95_ms": round(sorted(se)[int(len(se) * 0.95)], 1),
        "stab_e2e_max_ms": round(max(se), 1),
        "stab_decode_median": round(statistics.median(sd), 2),
        "stab_prompt_ms_median": round(statistics.median(spm), 1),
        "stab_unique_outputs": uniq,
    }
    with open(f"raw/nav_cache_{args.tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
