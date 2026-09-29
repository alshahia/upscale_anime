
import json, shutil, os
src = r"E:\python projects\upscale_anime\tmp\exa_search_results.json"
dst = r"E:\python projects\upscale_anime\docs\research\anime_sr_2026\exa_search_results.json"
shutil.copy2(src, dst)
print("Copied exa_search_results.json:", os.path.getsize(dst), "bytes")

# Generate SOURCES.md from the JSON
data = json.load(open(src, encoding="utf-8"))
print(f"Total queries: {len(data)}")
for d in data:
    print(f"  - {d.get('label')}: {len(d.get('results', []))} hits")
