#!/usr/bin/env python3
"""Ornith-1.5-9B v6 (llama.cpp) benchmark: cold start / prefill / decode / e2e.

Server must be managed by the systemd user unit `ornith-v6`.
The script restarts it once to measure cold start, then benchmarks over HTTP.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8080"
MODEL = Path("/home/jetson/local-llm/models/ornith-9b-v6/ornith-9b-v6-Q4_K_M.gguf")
SERVICE = "ornith-v6"
FOR_JETSON = Path("/home/jetson/Desktop/for_jetson")
sys.path.insert(0, str(FOR_JETSON))
from eval_jetson import build_raw_prompt  # noqa: E402

OUT = Path(__file__).with_name("ornith-v6-bench-results.json")


def post(path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def stream_completion(prompt: str, n_predict: int, stop: list[str] | None = None):
    """Return (ttft_s, total_s, content, timings)."""
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": True,
    }
    if stop:
        payload["stop"] = stop
    req = urllib.request.Request(
        BASE + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    timings = None
    with urllib.request.urlopen(req, timeout=600.0) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            piece = ev.get("content") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(piece)
            if ev.get("timings"):
                timings = ev["timings"]
            if ev.get("stop"):
                break
    total = time.perf_counter() - t0
    return (ttft, total, "".join(parts), timings)


def wait_health(timeout: float = 120.0, interval: float = 0.05) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=1) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError("server did not become healthy")


def evict_page_cache(path: Path) -> None:
    """Drop clean page-cache pages for the model file (no root needed)."""
    subprocess.run(["sync"], check=False)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def journal_tail(n: int = 120) -> str:
    p = subprocess.run(
        ["journalctl", "--user-unit", SERVICE, "-n", str(n), "--no-pager", "-o", "cat"],
        capture_output=True, text=True,
    )
    return p.stdout


ELAPSED_RE = None


def parse_elapsed(line: str) -> float | None:
    """Extract llama.cpp's `h.mm.ss.mmm` elapsed tag from a log line."""
    import re
    global ELAPSED_RE
    if ELAPSED_RE is None:
        ELAPSED_RE = re.compile(r"\b(\d{1,2})\.(\d{2})\.(\d{3})\.(\d{3})\b")
    m = ELAPSED_RE.search(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi * 60 + s + ms / 1000 + us / 1_000_000


def cold_start() -> dict:
    subprocess.run(["systemctl", "--user", "stop", SERVICE], check=False)
    time.sleep(1.0)
    evict_page_cache(MODEL)
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
    t0 = time.perf_counter()
    subprocess.run(["systemctl", "--user", "start", SERVICE], check=True)
    ready = wait_health()
    # The health endpoint answers as soon as the HTTP thread is up; the model
    # is already loaded before that (see journal "model loaded").
    load_s = None
    listen_s = None
    for _ in range(20):
        log = journal_tail(200)
        load_s = listen_s = None
        for line in log.splitlines():
            if "model loaded" in line:
                sec = parse_elapsed(line)
                if sec is not None:
                    load_s = sec
            elif "listening on" in line:
                sec = parse_elapsed(line)
                if sec is not None:
                    listen_s = sec
        if listen_s is not None:
            break
        time.sleep(0.25)
    # Cold first request right after start.
    user = "從檔案室經過警衛室走到會議室B，途中不要經過儲藏櫃"
    prompt = build_raw_prompt(user)
    ttft, total, content, timings = stream_completion(prompt, 32, ["\n"])
    return {
        "wall_to_health_s": ready,
        "log_model_loaded_s": load_s,
        "log_listening_s": listen_s,
        "free_before": free.strip().splitlines()[:2],
        "first_request": {
            "prompt": user,
            "ttft_s": ttft,
            "total_s": total,
            "content": content,
            "timings": timings,
        },
    }


FILLER = "沿著走廊前進會經過茶水間與樓梯間，請注意標示方向並保持通行順序。"


def filler_prompt(approx_tokens: int) -> str:
    uid = uuid.uuid4().hex[:8]
    target_chars = max(8, int(approx_tokens * 1.25))
    body = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    return f"測試編號{uid}。{body}"


def prefill_sweep() -> list[dict]:
    rows = []
    for approx in (128, 256, 512, 1024, 2048):
        prompt = filler_prompt(approx)
        t0 = time.perf_counter()
        d = post("/completion", {
            "prompt": prompt, "n_predict": 1, "temperature": 0.0,
            "top_k": 1, "cache_prompt": False,
        })
        total = time.perf_counter() - t0
        tm = d.get("timings") or {}
        rows.append({
            "target_tokens": approx,
            "prompt_n": tm.get("prompt_n"),
            "prompt_ms": tm.get("prompt_ms"),
            "prefill_tps": tm.get("prompt_per_second"),
            "client_total_ms": total * 1000,
        })
        print(f"  prefill ~{approx:4d} tok -> actual {tm.get('prompt_n')} tok, "
              f"{tm.get('prompt_per_second'):.1f} tok/s ({tm.get('prompt_ms'):.0f} ms)")
    return rows


def decode_runs(n: int = 3) -> list[dict]:
    rows = []
    for i in range(n):
        prompt = build_raw_prompt(f"從第{i + 1}號門走到停車場，途中經過收發室") + "\n"
        t0 = time.perf_counter()
        d = post("/completion", {
            "prompt": prompt, "n_predict": 128, "temperature": 0.0,
            "top_k": 1, "cache_prompt": False, "ignore_eos": True,
        })
        total = time.perf_counter() - t0
        tm = d.get("timings") or {}
        rows.append({
            "run": i + 1,
            "predicted_n": tm.get("predicted_n"),
            "predicted_ms": tm.get("predicted_ms"),
            "decode_tps": tm.get("predicted_per_second"),
            "client_total_ms": total * 1000,
        })
        print(f"  decode run {i + 1}: {tm.get('predicted_n')} tok, "
              f"{tm.get('predicted_per_second'):.2f} tok/s ({tm.get('predicted_ms'):.0f} ms)")
    return rows


def e2e_runs(limit: int = 10) -> list[dict]:
    rows = [json.loads(l) for l in (FOR_JETSON / "datasets/nav_testset.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = set()
    picked = []
    for r in rows:
        if r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
        picked.append(r)
        if len(picked) >= limit:
            break
    out = []
    for i, r in enumerate(picked, 1):
        prompt = build_raw_prompt(r["prompt"])
        ttft, total, content, tm = stream_completion(prompt, 64, ["\n"])
        tm = tm or {}
        out.append({
            "id": r["id"], "prompt": r["prompt"], "expected": r["expected"],
            "content": content,
            "ttft_s": ttft, "total_s": total,
            "prompt_n": tm.get("prompt_n"),
            "prefill_tps": tm.get("prompt_per_second"),
            "decode_tps": tm.get("predicted_per_second"),
        })
        print(f"  e2e {i:2d}/{len(picked)} {r['id']:<8} {total * 1000:6.0f} ms  "
              f"ttft {ttft * 1000:5.0f} ms  -> {content!r}")
    return out


def main() -> None:
    result: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    print("[1/5] cold start")
    result["cold_start"] = cold_start()
    cs = result["cold_start"]
    print(f"  health after {cs['wall_to_health_s']:.2f}s, "
          f"log listening at {cs['log_listening_s']}s, "
          f"first req ttft={cs['first_request']['ttft_s'] * 1000:.0f} ms "
          f"total={cs['first_request']['total_s'] * 1000:.0f} ms "
          f"-> {cs['first_request']['content']!r}")

    print("[2/5] prefill sweep")
    result["prefill"] = prefill_sweep()

    print("[3/5] decode")
    result["decode"] = decode_runs(3)

    print("[4/5] e2e nav prompts")
    result["e2e"] = e2e_runs(10)

    # summaries
    dec = [r["decode_tps"] for r in result["decode"] if r["decode_tps"]]
    e2e = [r["total_s"] * 1000 for r in result["e2e"]]
    ttfts = [r["ttft_s"] * 1000 for r in result["e2e"]]
    result["summary"] = {
        "decode_tps_median": statistics.median(dec) if dec else None,
        "e2e_ms_median": statistics.median(e2e) if e2e else None,
        "e2e_ms_p95": (statistics.quantiles(e2e, n=20)[18] if len(e2e) >= 20
                       else max(e2e) if e2e else None),
        "ttft_ms_median": statistics.median(ttfts) if ttfts else None,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[5/5] wrote {OUT}")
    print("summary:", json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
