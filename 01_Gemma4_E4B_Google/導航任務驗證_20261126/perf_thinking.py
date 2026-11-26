#!/usr/bin/env python3
"""量測 Gemma 4 E4B 在導航任務上的 prefill / decode / e2e，thinking ON vs OFF。
- raw /completion（無 chat template，等價 thinking off）
- /v1/chat/completions enable_thinking=false
- /v1/chat/completions enable_thinking=true
用 nav_testset 前 N 題的 prompt，unique 化避免 cache。
"""
import json, random, statistics, sys, time, urllib.request
from pathlib import Path

HOST = "127.0.0.1"; PORT = 8080
DATA = Path("/home/jetson/Desktop/for_jetson/datasets/nav_testset.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

def post(path, body):
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def raw_completion(prompt):
    r = post("/completion", {"prompt": prompt, "n_predict": 64, "temperature": 0.0,
        "top_k": 1, "stop": ["\n"], "cache_prompt": False, "stream": False})
    t = r.get("timings", {})
    return t

def chat(prompt, enable_thinking):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": 64, "cache_prompt": False, "stream": False}
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    r = post("/v1/chat/completions", body)
    return r.get("timings", {})

def agg(rows):
    def med(k): 
        v=[x[k] for x in rows if x.get(k)]; return statistics.median(v) if v else 0
    return {
        "n": len(rows),
        "prompt_n": med("prompt_n"),
        "prompt_ms": round(med("prompt_ms"),1),
        "prefill_tps": round(med("prompt_per_second"),1),
        "predicted_n": med("predicted_n"),
        "predicted_ms": round(med("predicted_ms"),1),
        "decode_tps": round(med("predicted_per_second"),1),
        "e2e_ms": round(med("prompt_ms")+med("predicted_ms"),1),
    }

rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()][:N]
# build prompts like eval (rules + shots) - reuse simple: just raw prompt with rules prefix
RULES = ("任務：從路線描述中抽取所有地點，依實際通行順序輸出。\n格式：以逗號分隔的單行清單。\n"
         "規則：\n1. 順序以實際通行為準，不是文字出現順序。\n2. 被否定或需避開的地點不輸出。\n"
         "3. 移動方式（走路、開車、上樓）不算地點。\n4. 保留原文用詞，不翻譯、不改寫。\n5. 沒有地點時只輸出（無）。\n")
SHOTS = ("從值班室到機車棚，途中經過茶水吧\n值班室,茶水吧,機車棚\n"
         "從會議室D到閱覽室，不要經過儲藏櫃\n會議室D,閱覽室\n\n")

def build(p): return RULES + SHOTS + p + "\n"

# warmup
raw_completion(build("暖機")); chat("暖機", False); chat("暖機", True)

res = {"raw_completion": [], "chat_nothink": [], "chat_think": []}
for r in rows:
    u = f" [u-{random.randint(0,10**12)}]"
    p = build(r["prompt"] + u)
    res["raw_completion"].append(raw_completion(p))
    res["chat_nothink"].append(chat(r["prompt"] + u, False))
    res["chat_think"].append(chat(r["prompt"] + u, True))

out = {k: agg(v) for k, v in res.items()}
print(json.dumps(out, ensure_ascii=False, indent=2))
Path("/tmp/gemma4_eval/perf_thinking.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))