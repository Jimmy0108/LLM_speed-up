#!/usr/bin/env python3
"""thinking OFF 長答 decode/E2E 掃描（n-max × MTP on/off）。"""
import json, time, urllib.request, random, statistics, sys
CHT="http://127.0.0.1:8080/v1/chat/completions"
def chat(p,mt=128,think=False):
    b={"messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":mt,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":think}}
    rq=urllib.request.Request(CHT,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter();r=json.loads(urllib.request.urlopen(rq,timeout=300).read())
    return (time.perf_counter()-t0)*1000, r.get("timings",{})
qs=["詳細解釋光合作用全過程，分步骤說明。",
    "為什麼天空是藍的？詳細分析瑞利散射。",
    "證明勾股定理，並用兩種不同方法說明。",
    "解釋量子糾纏的原理與應用，分步骤說明。",
    "描述法國大革命的三個主要階段。"]
n=int(sys.argv[1]) if len(sys.argv)>1 else 3
dec=[];e2e=[];acc=[];pn=[]
for q in qs:
    for k in range(n):
        w,t=chat(q+f" [u-{random.randint(0,10**9)}]",128,False)
        dec.append(t.get("predicted_per_second",0));e2e.append(w);pn.append(t.get("predicted_n",0))
        g=t.get("draft_n",0);a=t.get("draft_n_accepted",0)
        if g:acc.append(a/g)
print(f"decode中位={statistics.median(dec):.1f} 平均={statistics.mean(dec):.1f} 範圍={min(dec):.1f}-{max(dec):.1f} | "
      f"E2E={statistics.median(e2e):.0f}ms | accept={(statistics.mean(acc) if acc else 0):.3f} | pred_n={statistics.median(pn):.0f} | n={len(dec)}")