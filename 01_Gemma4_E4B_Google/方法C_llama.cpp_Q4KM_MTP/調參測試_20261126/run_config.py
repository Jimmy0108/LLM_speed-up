#!/usr/bin/env python3
"""
Gemma 4 E4B + MTP 調參測試 harness
- 每個 config 重啟 llama-server
- 用 unique prompt（避開 prefix cache）
- 收集 server timings：TTFT(prompt_ms)、decode(predicted_per_second)、draft 接受率
- 結果輸出 JSON + CSV
"""
import argparse, json, os, random, statistics, subprocess, sys, time

import requests

LLAMA = os.path.expanduser("~/local-llm/build-sm87/bin/llama-server")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUTDIR = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8080"


def find_gguf(repo_prefix, filename):
    root = os.path.join(HUB, repo_prefix)
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"{filename} under {root}")


BASE = find_gguf("models--unsloth--gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf")
MTP = find_gguf("models--ggml-org--gemma-4-E4B-it-GGUF", "mtp-gemma-4-E4B-it-Q4_0.gguf")


def kill_server():
    subprocess.run(["pkill", "-x", "llama-server"], check=False)
    time.sleep(3)


def start_server(extra, logpath):
    cmd = [LLAMA, "-m", BASE,
           "-ngl", "99", "-fa", "on", "-c", "4096", "-b", "512", "-ub", "256",
           "-ctk", "q8_0", "-ctv", "q8_0",
           "--host", "127.0.0.1", "--port", "8080"] + extra
    log = open(logpath, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    for _ in range(90):
        time.sleep(1)
        try:
            r = requests.get(URL + "/health", timeout=2)
            if r.ok:
                return proc, time.time() - t0
        except Exception:
            pass
    raise RuntimeError("server did not become healthy; see " + logpath)


def one_request(prompt, max_tokens=64):
    """Non-stream request; returns server-side timings (accurate, cache-safe if prompt unique)."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "cache_prompt": False,
    }
    t0 = time.perf_counter()
    r = requests.post(URL + "/v1/chat/completions", json=payload, timeout=300)
    wall = (time.perf_counter() - t0) * 1000
    d = r.json()
    t = d.get("timings", {})
    return {
        "wall_ms": wall,
        "prompt_n": t.get("prompt_n", 0),
        "prompt_ms": t.get("prompt_ms", 0),
        "prompt_per_second": t.get("prompt_per_second", 0),
        "predicted_n": t.get("predicted_n", 0),
        "predicted_ms": t.get("predicted_ms", 0),
        "predicted_per_second": t.get("predicted_per_second", 0),
        "draft_n": t.get("draft_n", 0),
        "draft_n_accepted": t.get("draft_n_accepted", 0),
    }


def warmup(n=3):
    for i in range(n):
        one_request(f"Say hello number {random.randint(0, 10**9)}.", 8)


def bench_decode(n_runs=10):
    rows = []
    for i in range(n_runs):
        uid = random.randint(0, 10**12)
        rows.append(one_request(f"Write exactly 50 words about the ocean. [req-{uid}]", 64))
    return rows


def bench_prefill():
    """Unique long prompts -> real prefill (cache miss)."""
    out = []
    for target in (256, 512, 1024):
        words = ["ocean", "mountain", "river", "forest", "desert", "island", "valley", "cloud"]
        # ~ target tokens: 1 word ~1.3 tok; build by repetition then unique suffix
        body = " ".join(random.choice(words) for _ in range(int(target / 1.3)))
        uid = random.randint(0, 10**12)
        prompt = body + f" [unique-{uid}] Summarize in one word."
        r = one_request(prompt, 1)
        out.append({"target": target, **r})
    return out


def bench_ngram(n_runs=5):
    """Highly repetitive prompt: n-gram spec should copy from prompt."""
    items = ", ".join(f"zone{i:03d}" for i in range(160))
    prompt = (
        "Copy the following list exactly, then continue with zone160, zone161, zone162:\n"
        + items
    )
    rows = []
    for i in range(n_runs):
        uid = random.randint(0, 10**12)
        rows.append(one_request(prompt + f" [u-{uid}]", 64))
    return rows


def summarize(rows, key="predicted_per_second"):
    vals = [r[key] for r in rows if r.get(key)]
    if not vals:
        return {}
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra", default="", help="extra server args (space separated)")
    ap.add_argument("--ngram-bench", action="store_true")
    args = ap.parse_args()

    extra = args.extra.split()
    logpath = os.path.join(OUTDIR, f"server-{args.name}.log")

    kill_server()
    proc, load_s = start_server(extra, logpath)
    try:
        warmup(3)
        dec = bench_decode(10)
        pre = bench_prefill()
        ng = bench_ngram(5) if args.ngram_bench else []

        acc = sum(r["draft_n_accepted"] for r in dec)
        gen = sum(r["draft_n"] for r in dec)
        result = {
            "name": args.name,
            "extra_args": extra,
            "server_load_s": round(load_s, 2),
            "decode": summarize(dec),
            "ttft_ms": summarize(dec, "prompt_ms"),
            "prompt_tps": summarize(dec, "prompt_per_second"),
            "draft": {
                "accepted": acc,
                "generated": gen,
                "acceptance_rate": (acc / gen) if gen else 0,
                "mean_len": (acc / len(dec)) if dec else 0,
            },
            "prefill": [
                {"target": p["target"], "prompt_n": p["prompt_n"],
                 "prompt_per_second": p["prompt_per_second"], "prompt_ms": p["prompt_ms"]}
                for p in pre
            ],
            "ngram": summarize(ng) if ng else None,
            "raw_decode": dec,
        }
        with open(os.path.join(OUTDIR, f"result-{args.name}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(json.dumps({k: result[k] for k in
                          ("name", "server_load_s", "decode", "ttft_ms", "draft")},
                         ensure_ascii=False, indent=2))
    finally:
        kill_server()


if __name__ == "__main__":
    main()
