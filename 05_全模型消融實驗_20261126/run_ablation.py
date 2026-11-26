#!/usr/bin/env python3
"""公平消融實驗 v2：所有 llama.cpp 模型 × MTP on/off，跑 for_jetson 導航測試。
統一：llama.cpp build-sm87, -m <model>, -c 4096 -t 6 -ngl 99 -fa on, KV q8_0。
收集：EM/F1, prefill/decode/e2e, RAM, 功耗(W), 溫度(°C), 載入時間(s), draft 接受率。
"""
import json, os, re, subprocess, time, statistics, urllib.request, random
from pathlib import Path

LLAMA = os.path.expanduser("~/local-llm/build-sm87/bin/llama-server")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
EVAL = "/home/jetson/Desktop/for_jetson/eval_jetson.py"
DATA = "/home/jetson/Desktop/for_jetson/datasets"
PY = os.path.expanduser("~/Desktop/TensorRT_Edge_LLM_Test/.venv-edgellm/bin/python")
OUT = Path("/tmp/ablation"); OUT.mkdir(exist_ok=True)
URL = "http://127.0.0.1:8080"

def find(repo, fn):
    for dp,_,fs in os.walk(os.path.join(HUB,repo)):
        if fn in fs: return os.path.join(dp,fn)
    raise FileNotFoundError(fn)

BASE = find("models--unsloth--gemma-4-E4B-it-GGUF","gemma-4-E4B-it-Q4_K_M.gguf")
Q4_0 = find("models--google--gemma-4-E4B-it-qat-q4_0-gguf","gemma-4-E4B_q4_0-it.gguf")
MTP  = find("models--ggml-org--gemma-4-E4B-it-GGUF","mtp-gemma-4-E4B-it-Q4_0.gguf")
ORN_V6 = os.path.expanduser("~/local-llm/models/ornith-9b-v6/ornith-9b-v6-Q4_K_M.gguf")
QWEN = os.path.expanduser("~/local-llm/models/Qwen3.5-4B-Q4_K_M.gguf")

CONFIGS = [
    ("gemma4_q4km_noMTP",  BASE,  []),
    ("gemma4_q4km_MTP",    BASE,  ["-md",MTP,"--spec-type","draft-mtp","--spec-draft-n-max","6"]),
    ("gemma4_q4_0_noMTP",  Q4_0,  []),
    ("ornith_v6_noMTP",    ORN_V6,[]),
    ("ornith_v6_MTP",      ORN_V6,["--spec-type","draft-mtp","--spec-draft-n-max","3"]),
    ("qwen35_4b_noMTP",    QWEN,  []),
]
COMMON = ["-ngl","99","-fa","on","-c","4096","-b","512","-ub","256","-t","6",
          "-ctk","q8_0","-ctv","q8_0","--host","127.0.0.1","--port","8080"]

def kill():
    subprocess.run(["pkill","-x","llama-server"],check=False); time.sleep(3)

def mem_used():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): avail=int(l.split()[1])
    return round((15533-avail//1024)/1024,2)

def sample_power_temp(sec=6):
    """跑 tegrastats sec 秒，回傳平均功耗(W)與 gpu 溫度(°C)。"""
    try:
        p=subprocess.Popen(["tegrastats"],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        time.sleep(sec); p.terminate()
        out=p.communicate(timeout=5)[0]
    except Exception:
        return None,None
    powers=[]; temps=[]
    for line in out.splitlines():
        m=re.search(r"VDD_IN (\d+)mW",line)
        if m: powers.append(int(m.group(1)))
        t=re.search(r"gpu@([\d.]+)C",line)
        if t: temps.append(float(t.group(1)))
    return (round(statistics.mean(powers)/1000,2) if powers else None,
            round(statistics.mean(temps),1) if temps else None)

def start(model, extra, log):
    cmd=[LLAMA,"-m",model]+COMMON+extra
    f=open(log,"w")
    t0=time.time()
    p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(120):
        time.sleep(1)
        try:
            if urllib.request.urlopen(URL+"/health",timeout=2).status==200:
                return p, round(time.time()-t0,1)
        except: pass
    raise RuntimeError("no health: "+log)

def run_eval(tag):
    subprocess.run([PY,EVAL,"--port","8080","--testset",f"{DATA}/nav_testset.jsonl",
        "--tag",tag+"_test","--out",f"{OUT}/{tag}_test.json"],capture_output=True,text=True,timeout=600)
    subprocess.run([PY,EVAL,"--port","8080","--testset",f"{DATA}/nav_holdout_v1.jsonl",
        "--tag",tag+"_hold","--out",f"{OUT}/{tag}_hold.json"],capture_output=True,text=True,timeout=600)
    return json.load(open(f"{OUT}/{tag}_test.json")), json.load(open(f"{OUT}/{tag}_hold.json"))

def perf(n=20):
    rows=[json.loads(l) for l in open(f"{DATA}/nav_testset.jsonl") if l.strip()][:n]
    RULES="任務：從路線描述中抽取所有地點，依實際通行順序輸出。\n格式：逗號分隔。\n\n"
    pre=[];tps=[];tot=[];acc=[]
    for r in rows:
        uid=random.randint(0,10**12)
        body=json.dumps({"prompt":RULES+r["prompt"]+f" [u-{uid}]\n","n_predict":32,
            "temperature":0,"top_k":1,"stop":["\n"],"cache_prompt":False}).encode()
        req=urllib.request.Request(URL+"/completion",data=body,headers={"Content-Type":"application/json"})
        t=json.loads(urllib.request.urlopen(req,timeout=120).read()).get("timings",{})
        pre.append(t.get("prompt_per_second",0)); tps.append(t.get("predicted_per_second",0))
        tot.append(t.get("prompt_ms",0)+t.get("predicted_ms",0))
        acc.append((t.get("draft_n_accepted",0),t.get("draft_n",0)))
    A=sum(x[0] for x in acc); G=sum(x[1] for x in acc)
    # 功耗/溫度在 perf 期間取樣
    pw,tp=sample_power_temp(6)
    return {"prefill_tps":round(statistics.median(pre),1),
            "decode_tps":round(statistics.median(tps),1),
            "e2e_ms":round(statistics.median(tot),0),
            "draft_acc":round(A/G,3) if G else None,
            "power_w":pw,"temp_c":tp}

def draft_from_log(log):
    txt=open(log).read()
    m=re.findall(r"draft acceptance = ([\d.]+)",txt)
    return round(float(m[-1]),3) if m else None

results=[]
for name,model,extra in CONFIGS:
    print(f"\n===== {name} =====")
    kill()
    log=f"{OUT}/server-{name}.log"
    p,load=start(model,extra,log)
    try:
        time.sleep(2)
        mem=mem_used()
        d,h=run_eval(name)
        pf=perf()
        results.append({"name":name,"model":os.path.basename(model),
            "mtp":bool(extra),"load_s":load,"ram_gb":mem,
            "test_em":round(d["em"],4),"test_f1":round(d["set_f1"],4),
            "hold_em":round(h["em"],4),"hold_f1":round(h["set_f1"],4),
            "test_ms_q":round(d["total_s"]/d["n"]*1000,0),
            "hold_ms_q":round(h["total_s"]/h["n"]*1000,0),
            **pf,"draft_log":draft_from_log(log)})
        r=results[-1]
        print(f"  load={load}s RAM={mem}GB power={pf['power_w']}W temp={pf['temp_c']}C")
        print(f"  test EM={d['em']:.3f} hold EM={h['em']:.3f} prefill={pf['prefill_tps']} decode={pf['decode_tps']} e2e={pf['e2e_ms']}ms draft={pf['draft_acc']}")
    finally:
        kill()

json.dump(results,open(f"{OUT}/summary.json","w"),ensure_ascii=False,indent=2)
print("\n===== 完成，摘要 =====")
print(json.dumps(results,ensure_ascii=False,indent=2))