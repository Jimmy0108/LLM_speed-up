#!/usr/bin/env python3
"""C1 A/B: 思考內容 decode + 短答 E2E，固定 vs adaptive。"""
import json, time, urllib.request, random, statistics, sys
CHT="http://127.0.0.1:8080/v1/chat/completions"
CURL="http://127.0.0.1:8080/completion"
def chat(p,mt,think=True):
    b={"messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":mt,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":think}}
    rq=urllib.request.Request(CHT,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter();r=json.loads(urllib.request.urlopen(rq,timeout=300).read())
    return (time.perf_counter()-t0)*1000, r.get("timings",{})
def raw(p,mt):
    b={"prompt":p,"n_predict":mt,"temperature":0,"top_k":1,"stop":["\n"],"cache_prompt":False}
    rq=urllib.request.Request(CURL,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter();r=json.loads(urllib.request.urlopen(rq,timeout=120).read())
    return (time.perf_counter()-t0)*1000, r.get("timings",{})
THINK=["一步一步推理：一個農場有雞和兔共35頭，94隻腳，請問雞和兔各幾隻？",
       "詳細解釋光合作用全過程，分步骤說明。","為什麼天空是藍的？詳細分析瑞利散射。"]
LITE="抽取路線中的所有地點，依實際通行順序以逗號分隔輸出。被否定或避開的地點不輸出；沒有地點時輸出（無）。\n"
NAV=["從大廳往會議室走，途中經過櫃檯","先去玄關，再去門廳，最後到天台","從停車場到辦公室，不要經過走廊"]
mode=sys.argv[1] if len(sys.argv)>1 else "think"
if mode=="think":
    dec=[];acc=[];e2e=[]
    for q in THINK:
        for k in range(4):
            w,t=chat(q+f" [u-{random.randint(0,10**9)}]",128)
            dec.append(t.get("predicted_per_second",0));e2e.append(w)
            g=t.get("draft_n",0);a=t.get("draft_n_accepted",0)
            if g:acc.append(a/g)
    print(f"THINK decode={statistics.median(dec):.1f} t/s  e2e={statistics.median(e2e):.0f}ms  accept={statistics.mean(acc):.3f}  n={len(dec)}")
elif mode=="short":
    e2e=[];dec=[]
    for q in NAV:
        w,t=raw(LITE+q+f" [u-{random.randint(0,10**9)}]\n",16)
        e2e.append(w);dec.append(t.get("predicted_per_second",0))
    print(f"SHORT e2e={statistics.median(e2e):.0f}ms  decode={statistics.median(dec):.1f}  n={len(e2e)}")
elif mode=="mixed":
    # 混合：3 思考 + 3 短答
    d1=[];d2=[]
    for q in THINK:
        w,t=chat(q+f" [u-{random.randint(0,10**9)}]",128);d1.append(t.get("predicted_per_second",0))
    for q in NAV:
        w,t=raw(LITE+q+f" [u-{random.randint(0,10**9)}]\n",16);d2.append(t.get("predicted_per_second",0))
    print(f"MIXED think={statistics.median(d1):.1f}  short={statistics.median(d2):.1f}")