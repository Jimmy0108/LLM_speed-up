#!/usr/bin/env python3
"""bench_dual.py — Gemma 4 E4B 雙目標量測 harness
- 支援 /completion（raw）與 /v1/chat/completions（thinking on/off）
- 每筆 unique prompt（cache_prompt=False）避開 prefix cache
- 收集 server timings：prefill / decode / E2E / draft 接受率
- 同時取樣 RAM / swap / power / temp（tegrastats）
- 輸出 runsheet.md（人看）+ runs.jsonl（AI 回朔）+ summary.json
- 每筆自動標註 IO/運算瓶頸
用法：
  python3 bench_dual.py --phase P0 --config base --matrix
  python3 bench_dual.py --phase P1 --config A1_pmin0.2 --p-min 0.2 --out 1,4,8
"""
import argparse, json, os, random, re, statistics, subprocess, time, urllib.request
from pathlib import Path

HOST, PORT = "127.0.0.1", 8080
URL = f"http://{HOST}:{PORT}"
OUTDIR = Path(os.path.dirname(os.path.abspath(__file__))) / "raw"
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA = Path("/home/jetson/Desktop/for_jetson/datasets/nav_testset.jsonl")

RULES = ("任務：從路線描述中抽取所有地點，依實際通行順序輸出。\n格式：以逗號分隔的單行清單。\n"
         "規則：\n1. 順序以實際通行為準，不是文字出現順序。\n2. 被否定或需避開的地點不輸出。\n"
         "3. 移動方式（走路、開車、上樓）不算地點。\n4. 保留原文用詞，不翻譯、不改寫。\n5. 沒有地點時只輸出（無）。\n")
SHOTS = ("從值班室到機車棚，途中經過茶水吧\n值班室,茶水吧,機車棚\n"
         "從會議室D到閱覽室，不要經過儲藏櫃\n會議室D,閱覽室\n\n")

def post(path, body, timeout=300):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def mem_swap():
    mem = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":", 1)
        mem[k] = int(v.split()[0])
    swap_used = 0
    try:
        for l in open("/proc/swaps"):
            if l.startswith("/"):
                p = l.split()
                swap_used += int(p[2]) - int(p[3])
    except Exception:
        pass
    return mem.get("MemAvailable", 0) // 1024, swap_used // 1024  # MB

def sample_env(sec=2.5):
    """取 tegrastats 數秒，回 power_mW / temp_c / gr3d。"""
    try:
        p = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
        time.sleep(sec); p.terminate()
        out = p.communicate(timeout=5)[0]
    except Exception:
        return None, None, None
    pw = [int(m.group(1)) for l in out.splitlines() if (m := re.search(r"VDD_IN (\d+)mW", l))]
    tp = [float(m.group(1)) for l in out.splitlines() if (m := re.search(r"gpu@([\d.]+)C", l))]
    gr = [int(m.group(1)) for l in out.splitlines() if (m := re.search(r"GR3D_FREQ (\d+)%", l))]
    return (round(statistics.mean(pw)) if pw else None,
            round(statistics.mean(tp), 1) if tp else None,
            round(statistics.mean(gr)) if gr else None)

def bg_load():
    """檢查真正的重背景程序（chrome / system-monitor / update-manager）數量。"""
    heavy = ("chrome", "gnome-system-monitor", "update-manager", "tracker-extract")
    n = 0
    try:
        out = subprocess.run(["ps", "-eo", "comm"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            c = line.strip()
            if any(h in c for h in heavy):
                n += 1
    except Exception:
        pass
    return n

def classify(prefill_tps, decode_tps, prompt_n, swap_used, bg_running, temp, draft_overhead, accept):
    """自動標註 IO/運算瓶頸。bg_running=非 llama 的 running 程序數。"""
    tags = []
    if swap_used > 50:
        tags.append("IO_SWAP")
    if bg_running > 2:
        tags.append("BACKGROUND")
    if temp and temp > 80:
        tags.append("THERMAL")
    if decode_tps and decode_tps < 15 and prompt_n and prompt_n > 200:
        tags.append("MEM_BOUND")
    if prefill_tps and prompt_n and prefill_tps < 300 and prompt_n > 200:
        tags.append("COMPUTE_BOUND")
    if draft_overhead and draft_overhead > 0.20 and accept is not None and accept < 0.2:
        tags.append("SYNC_OVERHEAD")
    return ",".join(tags) if tags else "OK"

def one_raw(prompt, n_predict, stop=True):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "cache_prompt": False, "stream": False}
    if stop:
        body["stop"] = ["\n"]
    t0 = time.perf_counter()
    r = post("/completion", body)
    wall = (time.perf_counter() - t0) * 1000
    t = r.get("timings", {})
    return {
        "wall_ms": round(wall, 1),
        "prompt_n": t.get("prompt_n", 0), "prompt_ms": round(t.get("prompt_ms", 0), 1),
        "prefill_tps": round(t.get("prompt_per_second", 0), 1),
        "predicted_n": t.get("predicted_n", 0), "predicted_ms": round(t.get("predicted_ms", 0), 1),
        "decode_tps": round(t.get("predicted_per_second", 0), 1),
        "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
        "content": r.get("content", "")[:80],
    }

def one_chat(prompt, max_tokens, thinking):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": max_tokens, "cache_prompt": False, "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    t0 = time.perf_counter()
    r = post("/v1/chat/completions", body)
    wall = (time.perf_counter() - t0) * 1000
    t = r.get("timings", {})
    return {
        "wall_ms": round(wall, 1),
        "prompt_n": t.get("prompt_n", 0), "prompt_ms": round(t.get("prompt_ms", 0), 1),
        "prefill_tps": round(t.get("prompt_per_second", 0), 1),
        "predicted_n": t.get("predicted_n", 0), "predicted_ms": round(t.get("predicted_ms", 0), 1),
        "decode_tps": round(t.get("predicted_per_second", 0), 1),
        "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
        "content": r.get("choices", [{}])[0].get("message", {}).get("content", "")[:80],
    }

def nav_prompt(i):
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = rows[i % len(rows)]
    return RULES + SHOTS + r["prompt"] + f" [u-{random.randint(0,10**12)}]\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--matrix", action="store_true", help="(in,out) 矩陣")
    ap.add_argument("--n", type=int, default=10, help="每格重複次數")
    ap.add_argument("--out", default="", help="逗號分隔 out 值（短答用）")
    ap.add_argument("--thinking", default="off", choices=["on", "off", "both"])
    ap.add_argument("--ignore-eos", action="store_true", help="長文固定 decode")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    load = float(open("/proc/loadavg").read().split()[0])
    bg = bg_load()
    avail_mb, swap_mb = mem_swap()
    power, temp, gr3d = sample_env(2.5)

    runs = []
    def run(tag, fn):
        for k in range(args.n):
            r = fn(k)
            r.update({"request_id": f"{args.phase}-{args.config}-{tag}-{k}",
                      "phase": args.phase, "config": args.config, "tag": tag,
                      "load": round(load, 2), "bg_running": bg, "ram_avail_mb": avail_mb,
                      "swap_used_mb": swap_mb, "power_mw": power, "temp_c": temp, "gr3d": gr3d})
            gen = r.get("draft_n", 0); acc = r.get("draft_accepted", 0)
            r["accept_rate"] = round(acc / gen, 3) if gen else None
            r["io_compute_note"] = classify(r["prefill_tps"], r["decode_tps"], r["prompt_n"],
                                            swap_mb, bg, temp,
                                            (gen - acc) / gen if gen else None, r["accept_rate"])
            runs.append(r)
            print(f"  {r['request_id']:<28} pre={r['prefill_tps']:>6} dec={r['decode_tps']:>6} "
                  f"e2e={r['wall_ms']:>7.0f}ms draft={acc}/{gen} [{r['io_compute_note']}]")

    if args.matrix:
        for nin in (32, 128, 512):
            for nout in (1, 4, 8, 32, 128):
                run(f"in{nin}_out{nout}", lambda k, a=nin, b=nout: one_raw(
                    nav_prompt(k) if a < 500 else f"寫一段長文，主題是海洋 [u-{random.randint(0,10**12)}]",
                    b, stop=(b <= 8)))
    elif args.out:
        for o in [int(x) for x in args.out.split(",")]:
            run(f"out{o}", lambda k, b=o: one_raw(nav_prompt(k), b, stop=(b <= 8)))
    else:
        run("nav", lambda k: one_raw(nav_prompt(k), 32))

    # 寫 runs.jsonl
    jf = OUTDIR / f"{args.phase}_{args.config}_runs.jsonl"
    with open(jf, "w") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    def med(key):
        v = [r[key] for r in runs if r.get(key)]
        return round(statistics.median(v), 1) if v else 0
    summ = {"phase": args.phase, "config": args.config, "n": len(runs),
            "note": args.note, "env": {"load": round(load, 2), "bg_running": bg,
            "ram_avail_mb": avail_mb, "swap_used_mb": swap_mb,
            "power_mw": power, "temp_c": temp, "gr3d": gr3d},
            "prefill_tps": med("prefill_tps"), "decode_tps": med("decode_tps"),
            "e2e_ms": med("wall_ms"), "ttft_ms": med("prompt_ms"),
            "accept_rate": (round(sum(r["draft_accepted"] for r in runs) /
                                  max(1, sum(r["draft_n"] for r in runs)), 3))}
    (OUTDIR / f"{args.phase}_{args.config}_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2))

    # runsheet.md（人看）
    rs = OUTDIR / "runsheet.md"
    with open(rs, "a") as f:
        f.write(f"\n## {args.phase} / {args.config}  ({time.strftime('%H:%M:%S')})\n\n")
        f.write(f"env: load={load:.2f} bg={bg} ram_avail={avail_mb}MB swap={swap_mb}MB "
                f"power={power}mW temp={temp}C gr3d={gr3d}%\n\n")
        f.write("| request | prefill | decode | e2e_ms | ttft | draft | note |\n|---|---:|---:|---:|---:|---:|---|\n")
        for r in runs:
            f.write(f"| {r['request_id']} | {r['prefill_tps']} | {r['decode_tps']} | "
                    f"{r['wall_ms']:.0f} | {r['prompt_ms']:.0f} | "
                    f"{r['draft_accepted']}/{r['draft_n']} | {r['io_compute_note']} |\n")
    print(f"\n→ {jf}\n→ {OUTDIR}/runsheet.md\n→ summary: {summ}")

if __name__ == "__main__":
    main()