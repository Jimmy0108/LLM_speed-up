#!/usr/bin/env python3
"""JSON 結構化 thinking-OFF decode 掃描。"""
import json,time,urllib.request,random,statistics
CHT="http://127.0.0.1:8080/v1/chat/completions"
def chat(p,mt=128):
    b={"messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":mt,"cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False}}
    rq=urllib.request.Request(CHT,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter();r=json.loads(urllib.request.urlopen(rq,timeout=300).read())
    return (time.perf_counter()-t0)*1000,r.get("timings",{})
JSON=["用 JSON 陣列列出三個常見水果，每個含 name 與 color 欄位。",
      "用 JSON 陣列列出三個程式語言，每個含 name 與 year 欄位。",
      "用 JSON 陣列列出三個行星，每個含 name 與 order 欄位。",
      "用 JSON 陣列列出四個 ISO 國家代碼，每個含 code 與 capital。",
      "用 JSON 陣列列出三個資料結構，每個含 name 與 complexity。"]
dec=[];acc=[];e2e=[]
for q in JSON:
  for k in range(3):
    w,t=chat(q+f" [u-{random.randint(0,10**9)}]")
    dec.append(t.get("predicted_per_second",0));e2e.append(w);g=t.get("draft_n",0);a=t.get("draft_n_accepted",0)
    if g:acc.append(a/g)
print(f"decode中位={statistics.median(dec):.1f} accept={statistics.mean(acc):.3f} E2E={statistics.median(e2e):.0f}ms")