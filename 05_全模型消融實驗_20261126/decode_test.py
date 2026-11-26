#!/usr/bin/env python3
"""公平 decode 測試：每個配置強制產生固定 64 token（ignore_eos、無 stop），
隔離純 decode 速度；並在滿載下取功耗/溫度。與 eval 的短輸出解耦。"""
import json, os, re, subprocess, time, statistics, urllib.request, random
from pathlib import Path
LLAMA=os.path.expanduser("~/local-llm/build-sm87/bin/llama-server")
HUB=os.path.expanduser("~/.cache/huggingface/hub")
OUT=Path("/tmp/ablation"); URL="http://127.0.0.1:8080"
def find(repo,fn):
    for dp,_,fs in os.walk(os.path.join(HUB,repo)):
        if fn in fs: return os.path.join(dp,fn)
BASE=find("models--unsloth--gemma-4-E4B-it-GGUF","gemma-4-E4B-it-Q4_K_M.gguf")
Q4_0=find("models--google--gemma-4-E4B-it-qat-q4_0-gguf","gemma-4-E4B_q4_0-it.gguf")
MTP=find("models--ggml-org--gemma-4-E4B-it-GGUF","mtp-gemma-4-E4B-it-Q4_0.gguf")
ORN=os.path.expanduser("~/local-llm/models/ornith-9b-v6/ornith-9b-v6-Q4_K_M.gguf")
QWEN=os.path.expanduser("~/local-llm/models/Qwen3.5-4B-Q4_K_M.gguf")
CONFIGS=[("gemma4_q4km_noMTP",BASE,[]),
 ("gemma4_q4km_MTP",BASE,["-md",MTP,"--spec-type","draft-mtp","--spec-draft-n-max","6"]),
 ("gemma4_q4_0_noMTP",Q4_0,[]),
 ("ornith_v6_noMTP",ORN,[]),
 ("ornith_v6_MTP",ORN,["--spec-type","draft-mtp","--spec-draft-n-max","3"]),
 ("qwen35_4b_noMTP",QWEN,[])]
COMMON=["-ngl","99","-fa","on","-c","4096","-b","512","-ub","256","-t","6","-ctk","q8_0","-ctv","q8_0","--host","127.0.0.1","--port","8080"]
def kill(): subprocess.run(["pkill","-x","llama-server"],check=False); time.sleep(3)
def start(model,extra,log):
    f=open(log,"w"); p=subprocess.Popen([LLAMA,"-m",model]+COMMON+extra,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(120):
        time.sleep(1)
        try:
            if urllib.request.urlopen(URL+"/health",timeout=2).status==200: return p
        except: pass
    raise RuntimeError(log)
def power(sec=8):
    try:
        p=subprocess.Popen(["tegrastats"],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        time.sleep(sec); p.terminate(); out=p.communicate(timeout=5)[0]
    except: return None,None
    pw=[int(m.group(1)) for l in out.splitlines() if (m:=re.search(r"VDD_IN (\d+)mW",l))]
    tp=[float(m.group(1)) for l in out.splitlines() if (m:=re.search(r"gpu@([\d.]+)C",l))]
    return (round(statistics.mean(pw)/1000,2) if pw else None, round(statistics.mean(tp),1) if tp else None)
def decode_fixed(ntok=64):
    tps=[];acc=[]
    for i in range(5):
        uid=random.randint(0,10**12)
        body=json.dumps({"prompt":f"寫一段長文，主題是海洋 [u-{uid}]","n_predict":ntok,
            "temperature":0,"top_k":1,"cache_prompt":False,"ignore_eos":True}).encode()
        req=urllib.request.Request(URL+"/completion",data=body,headers={"Content-Type":"application/json"})
        t=json.loads(urllib.request.urlopen(req,timeout=120).read()).get("timings",{})
        tps.append(t.get("predicted_per_second",0)); acc.append((t.get("draft_n_accepted",0),t.get("draft_n",0)))
    A=sum(x[0] for x in acc);G=sum(x[1] for x in acc)
    return round(statistics.median(tps),1),(round(A/G,3) if G else None)
res=[]
for name,model,extra in CONFIGS:
    kill(); p=start(model,extra,f"{OUT}/dec-{name}.log")
    try:
        time.sleep(2)
        tps,acc=decode_fixed()
        pw,tp=power(8)
        res.append({"name":name,"decode_fixed64_tps":tps,"draft_acc":acc,"power_w":pw,"temp_c":tp})
        print(f"{name:<22} decode(64tok)={tps:>6} t/s  draft={acc}  power={pw}W  temp={tp}C")
    finally: kill()
json.dump(res,open(f"{OUT}/decode_summary.json","w"),ensure_ascii=False,indent=2)
print("done")