import sys, os, json, time
sys.path.insert(0, r"E:\python projects\upscale_anime\tmp")
from exa_helper import exa_research

SYS_PROMPT = """Prefer 2024-2026 sources. Include paper URL, repo URL, license, year, parameter count, latency, and quality metrics (PSNR, LPIPS, MANIQA, CLIPIQA) when available. For Japanese or Chinese sources, translate titles to English. Be concise and structured."""

queries = json.load(open(r"E:\python projects\upscale_anime\tmp\exa_queries.json", encoding="utf-8"))

results = {}
for q in queries:
    label = q["label"]
    print(f"\n=== {label} ===", flush=True)
    print(f"Query: {q[\"query\"][:100]}...", flush=True)
    try:
        t0 = time.time()
        r = exa_research(
            query=q["query"],
            schema=q["schema"],
            system_prompt=SYS_PROMPT,
            timeout_ms=300000,
        )
        results[label] = r
        cost = r.get("cost_dollars", {}).get("total", "?")
        print(f"OK in {time.time()-t0:.1f}s, cost ${cost}", flush=True)
        out_path = rf"E:\python projects\upscale_anime\tmp\exa_result_{label}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}", flush=True)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", flush=True)
        results[label] = {"error": str(exc)}

combined = r"E:\python projects\upscale_anime\tmp\exa_results_all.json"
with open(combined, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nALL DONE. Combined: {combined}", flush=True)
print(f"Total queries: {len(queries)}, OK: {sum(1 for v in results.values() if \"output\" in v)}", flush=True)