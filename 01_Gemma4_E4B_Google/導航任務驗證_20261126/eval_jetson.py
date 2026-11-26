#!/usr/bin/env python3
"""Jetson 用的自包含評估腳本（無需 openvino / 無需專案其他檔案）。

用法
----
    # 1. 啟動 llama.cpp server（在 model/ 旁）
    ./llama-server -m model/ornith-9b-v6-Q4_K_M.gguf -c 2048 --temp 0 -ngl 99

    # 2. 評估
    python3 eval_jetson.py --port 8080 --testset datasets/nav_testset.jsonl

    # 只跑 holdout
    python3 eval_jetson.py --port 8080 --testset datasets/nav_holdout_v1.jsonl

注意
----
**必須用 /completion，不要用 /v1/chat/completions。**
Ornith-1.5-9B 是 reasoning model，chat 端點會套 chat template 並插入
`<think>`，而我們的 LoRA 是 raw completion 格式訓練的。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt 建構（與訓練時完全一致，刻意 inline 以免依賴專案檔案）
# ---------------------------------------------------------------------------

RAW_RULES_ZH = (
    "任務：從路線描述中抽取所有地點，依實際通行順序輸出。\n"
    "格式：以逗號分隔的單行清單。\n"
    "規則：\n"
    "1. 順序以實際通行為準，不是文字出現順序。\n"
    "2. 被否定或需避開的地點不輸出。\n"
    "3. 移動方式（走路、開車、上樓）不算地點。\n"
    "4. 保留原文用詞，不翻譯、不改寫。\n"
    "5. 沒有地點時只輸出（無）。\n"
)
RAW_RULES_EN = (
    "Task: extract all places from the route description in actual travel order.\n"
    "Format: a single comma-separated line.\n"
    "Rules:\n"
    "1. Order follows actual travel, not text appearance.\n"
    "2. Skip negated or avoided places.\n"
    "3. Movement methods (walking, driving, going up) are not places.\n"
    "4. Keep original wording, do not translate.\n"
    "5. If there is no place, output only (none).\n"
)
RAW_RULES_LITE_ZH = (
    "抽取路線中的所有地點，依實際通行順序以逗號分隔輸出。"
    "被否定或避開的地點不輸出；沒有地點時輸出（無）。\n"
)
RAW_RULES_LITE_EN = (
    "Extract all places in actual travel order, comma separated. "
    "Skip negated or avoided places; output (none) if there is no place.\n"
)

#: few-shot 刻意使用與測試集不重疊的地點，並補齊樓層 / 混合 / 英文否定。
RAW_SHOTS_ZH = [
    ("從值班室到機車棚，途中經過茶水吧", "值班室,茶水吧,機車棚"),
    ("從會議室D到閱覽室，不要經過儲藏櫃", "會議室D,閱覽室"),
    ("從3樓機房通道到8樓觀景台，中間經過5樓茶水吧", "3樓機房通道,5樓茶水吧,8樓觀景台"),
    ("先到收發室，再去樓梯間，最後到頂樓陽台", "收發室,樓梯間,頂樓陽台"),
    ("從攝影棚到器材庫", "攝影棚,器材庫"),
    ("現在幾點？", "（無）"),
]
RAW_SHOTS_MIX = [
    ("從 parking 區到會議室D，途中經過 茶水吧", "parking 區,茶水吧,會議室D"),
    ("先到 mailroom，再去樓梯間", "mailroom,樓梯間"),
]
RAW_SHOTS_EN = [
    ("From the mailroom to the rooftop, passing the tea bar", "mailroom,tea bar,rooftop"),
    ("From the file room to meeting room D, avoid the storage cabinet", "file room,meeting room D"),
    ("First the guard post, then the loading dock, finally the warehouse", "guard post,loading dock,warehouse"),
    ("What time is it?", "(none)"),
]

#: few-shot 的嵌套子集優先序：小 N 時保留最關鍵的範式。
SHOT_PRIORITY_ZH = [1, 5, 0, 3, 2, 4]
SHOT_PRIORITY_EN = [1, 3, 0, 2]
SHOT_PRIORITY_MIX = [0, 1]

EMPTY_TOKENS = frozenset({"（無）", "(none)", "無", ""})


def _is_english(s: str) -> bool:
    letters = sum(c.isascii() and c.isalpha() for c in s)
    cjk = sum("\u4e00" <= c <= "\u9fff" for c in s)
    return letters > cjk


def _is_mixed(s: str) -> bool:
    letters = sum(c.isascii() and c.isalpha() for c in s)
    cjk = sum("\u4e00" <= c <= "\u9fff" for c in s)
    return letters >= 2 and cjk > 0 and letters <= cjk * 2


def _pick(shots: list, priority: list[int], n: int | None) -> list:
    if n is None:
        return list(shots)
    n = max(0, int(n))
    return [shots[i] for i in priority[:n] if i < len(shots)]


def build_raw_prompt(user: str, shots: int | None = None,
                     rules_mode: str = "full") -> str:
    """建構與 LoRA 訓練時完全一致的 raw completion prompt。"""
    if rules_mode == "auto":
        if _is_english(user):
            rules_mode = "lite"
        elif _is_mixed(user):
            rules_mode = "full"
        else:
            rules_mode = "none"

    if _is_english(user):
        group, prio = RAW_SHOTS_EN, SHOT_PRIORITY_EN
        rules = RAW_RULES_EN if rules_mode == "full" else (
            RAW_RULES_LITE_EN if rules_mode == "lite" else "")
    elif _is_mixed(user):
        group, prio = RAW_SHOTS_ZH[:4] + RAW_SHOTS_MIX, SHOT_PRIORITY_MIX
        rules = RAW_RULES_ZH if rules_mode == "full" else (
            RAW_RULES_LITE_ZH if rules_mode == "lite" else "")
    else:
        group, prio = RAW_SHOTS_ZH, SHOT_PRIORITY_ZH
        rules = RAW_RULES_ZH if rules_mode == "full" else (
            RAW_RULES_LITE_ZH if rules_mode == "lite" else "")

    picked = _pick(group, prio, shots)
    body = "\n".join(f"{q}\n{a}" for q, a in picked)
    if rules:
        return f"{rules}\n{body}\n\n{user}\n" if body else f"{rules}\n{user}\n"
    return f"{body}\n\n{user}\n" if body else f"{user}\n"


# ---------------------------------------------------------------------------
# 評分
# ---------------------------------------------------------------------------

def parse_tokens(raw: str) -> list[str]:
    """把模型輸出切成地名清單（去空白、轉小寫、丟掉「無」）。"""
    out: list[str] = []
    for part in raw.replace("，", ",").split(","):
        tok = part.strip().lower()
        if tok and tok not in EMPTY_TOKENS:
            out.append(tok)
    return out


def set_scores(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    """回傳 (precision, recall, f1)。空對空視為全對。"""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    p_set, g_set = set(pred), set(gold)
    hit = len(p_set & g_set)
    precision = hit / len(p_set) if p_set else 0.0
    recall = hit / len(g_set) if g_set else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def complete(url: str, prompt: str, max_tokens: int,
             timeout: float = 180.0) -> dict:
    """呼叫 llama.cpp 的 /completion 端點。"""
    body = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "stop": ["\n"],
        "cache_prompt": False,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="用 llama.cpp /completion 評估 Ornith-9B v6 GGUF")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--testset", default="datasets/nav_testset.jsonl")
    ap.add_argument("--tag", default="jetson")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--rules", default="full",
                    choices=["full", "lite", "none", "auto"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.testset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    url = f"http://{args.host}:{args.port}/completion"
    print(f"[info] {url}  items={len(rows)}  rules={args.rules}")

    try:
        t0 = time.perf_counter()
        complete(url, "暖機", 4)
        print(f"[info] warmup {time.perf_counter() - t0:.2f}s")
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"無法連線到 {url}\n"
            f"先啟動 server：\n"
            f"  ./llama-server -m model/ornith-9b-v6-Q4_K_M.gguf "
            f"-c 2048 --temp 0 -ngl 99\n"
            f"錯誤：{exc}"
        )

    items: list[dict] = []
    t_start = time.perf_counter()
    for i, r in enumerate(rows, 1):
        prompt = build_raw_prompt(r["prompt"], rules_mode=args.rules)
        t0 = time.perf_counter()
        try:
            resp = complete(url, prompt, args.max_new_tokens)
            raw = resp.get("content", "").strip()
        except (urllib.error.URLError, TimeoutError) as exc:
            raw = f"<ERROR: {exc}>"
        e2e = time.perf_counter() - t0

        pred = parse_tokens(raw)
        gold = [str(g).strip().lower() for g in r["expected"]]
        precision, recall, f1 = set_scores(pred, gold)
        items.append({
            "id": r["id"], "cat": r.get("category", "?"),
            "prompt": r["prompt"], "raw": raw, "pred": pred, "gold": gold,
            "em": pred == gold, "f1": f1, "e2e_s": e2e,
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  {time.perf_counter() - t_start:.1f}s")

    total = time.perf_counter() - t_start
    n = len(items)
    em = sum(x["em"] for x in items) / n
    f1 = sum(x["f1"] for x in items) / n

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for x in items:
        by_cat[x["cat"]].append(x)

    print(f"\n=== {args.tag} ===")
    print(f"  EM={em:.4f}  F1={f1:.4f}  n={n}  "
          f"{total:.1f}s ({total / n * 1000:.0f} ms/題)")
    print(f"  {'類別':<20}{'n':>4}{'EM':>8}")
    for cat in sorted(by_cat):
        rc = by_cat[cat]
        print(f"  {cat:<20}{len(rc):>4}"
              f"{sum(x['em'] for x in rc) / len(rc):>8.4f}")

    bad = [x for x in items if not x["em"]]
    if bad:
        print(f"\n--- 錯誤 {len(bad)} 題 ---")
        for x in bad[:15]:
            print(f"  {x['id']:<9} gold={x['gold']}")
            print(f"  {'':<9} raw ={x['raw']!r}")

    out = Path(args.out) if args.out else Path(f"results_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag, "testset": args.testset, "rules": args.rules,
        "n": n, "em": em, "set_f1": f1, "total_s": total,
        "by_cat": {c: {"n": len(v),
                       "em": sum(x["em"] for x in v) / len(v)}
                   for c, v in sorted(by_cat.items())},
        "items": items,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫入 {out}")


if __name__ == "__main__":
    main()