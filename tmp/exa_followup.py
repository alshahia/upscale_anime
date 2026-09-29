import os, json, time, asyncio
import exa_py

exa = exa_py.Exa("0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")

QUERIES = [
    ("h", "y"),
    ("S", "e"),
    ("s", "u"),
]

def safe(obj):
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, (list, tuple)): return [safe(x) for x in obj]
    if isinstance(obj, dict): return {str(k): safe(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"): return safe(obj.model_dump())
    if hasattr(obj, "__dict__"): return safe(vars(obj))
    return str(obj)

async def run_one(label, q):
    t0 = time.time()
    try:
        r = exa.search(q, num_results=8, type="auto", contents={"highlights": {"numSentences": 4, "highlightsPerUrl": 3}})
        out = safe({
            "label": label, "query": q,
            "resolved_search_type": r.resolved_search_type,
            "search_time_ms": r.search_time,
            "cost_dollars": r.cost_dollars,
            "results": [
                {"title": x.title, "url": x.url, "published_date": str(getattr(x, "published_date", "") or ""), "author": getattr(x, "author", None), "highlights": (x.highlights or [])[:3]}
                for x in r.results
            ],
        })
        cost = float(getattr(r.cost_dollars, "total", 0) or 0)
        print("[" + label + "] " + str(round(time.time()-t0, 1)) + "s $" + str(cost) + " " + str(len(r.results)) + " hits", flush=True)
        return out
    except Exception as exc:
        print("[" + label + "] FAIL " + type(exc).__name__ + ": " + str(exc), flush=True)
        return {"label": label, "query": q, "error": str(exc)}

async def main():
    tasks = [run_one(lbl, q) for lbl, q in QUERIES]
    results = await asyncio.gather(*tasks)
    # Append to existing results file
    base = "E:\\python projects\\upscale_anime\\tmp\\exa_search_results.json"
    existing = []
    if os.path.exists(base):
        existing = json.load(open(base, encoding="utf-8"))
    existing_labels = {r["label"] for r in existing}
    for r in results:
        if r.get("label") not in existing_labels:
            existing.append(r)
    with open(base, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    total = 0.0
    for r in results:
        c = r.get("cost_dollars") or {}
        if isinstance(c, dict):
            total = total + float(c.get("total", 0) or 0)
    print("DONE. " + str(len(results)) + " follow-ups, $" + str(total), flush=True)

asyncio.run(main())