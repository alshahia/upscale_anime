
import os, json, time, asyncio
import exa_py
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

exa = exa_py.Exa("0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")

QUERIES = [
    ("hybrid_cascade_2x4x", "hybrid super resolution cascade 2x 2x anime lightweight student + diffusion post-process 2024 2025 2026 real-time achievable"),
    ("diffusion_anime_finetune", "SeeSR SUPIR anime fine-tuned anime-specific diffusion super resolution 2025 cartoon illustration dataset results"),
    ("sota_2025_2026_sr", "super resolution state of the art 2025 2026 latest publication OneRestore DiffIRv2 HAT-Lite RealPLKSR DAT2 anime quality"),
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
        return safe({
            "label": label, "query": q,
            "resolved_search_type": r.resolved_search_type,
            "search_time_ms": r.search_time,
            "cost_dollars": r.cost_dollars,
            "results": [
                {"title": x.title, "url": x.url, "published_date": str(getattr(x, "published_date", "") or ""), "author": getattr(x, "author", None), "highlights": (x.highlights or [])[:3]}
                for x in r.results
            ],
        })
    except Exception as exc:
        return {"label": label, "query": q, "error": str(exc)}

async def main():
    results = await asyncio.gather(*[run_one(l, q) for l, q in QUERIES])
    base = "E:\\python projects\\upscale_anime\\tmp\\exa_search_results.json"
    existing = json.load(open(base, encoding="utf-8"))
    # Replace the 3 mangled entries with proper ones
    new_labels = {r["label"] for r in results}
    existing = [e for e in existing if e.get("label") not in new_labels]
    existing.extend(results)
    with open(base, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"Saved. Total: {len(existing)} queries. New follow-ups: {len(results)}")
    # Dump highlights of new ones to file
    out = []
    for d in results:
        out.append(f"\n=== {d['label']} ===")
        out.append(f"  query: {d['query']}")
        for i, r in enumerate(d.get("results", [])[:5]):
            out.append(f"  [{i}] {r.get('title','')}")
            out.append(f"      url: {r.get('url','')}")
            for h in (r.get("highlights") or [])[:1]:
                cleaned = h[:300].replace(chr(10), " ").replace(chr(13), " ")
                out.append(f"      HL: {cleaned}")
    with open(r"E:\python projects\upscale_anime\tmp\followup_dump.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("Dumped to followup_dump.txt:", len(out), "lines")

asyncio.run(main())
