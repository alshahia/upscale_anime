import json
data = json.load(open(r"E:\python projects\upscale_anime\tmp\exa_search_results.json", encoding="utf-8"))
print("Total queries:", len(data))
print("Labels:", [d.get("label") for d in data])
print()
for d in data:
    print(f"\n=== {d.get('label')} ===")
    print(f"Query: {d.get('query', '?')[:100]}...")
    if "results" in d:
        for i, r in enumerate(d["results"][:5]):
            print(f"  [{i}] {r.get('title','?')[:80]}")
            print(f"      {r.get('url','?')[:90]}")
            for h in (r.get("highlights") or [])[:1]:
                print(f"      HL: {h[:250].replace(chr(10),' ')}")
