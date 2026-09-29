import json, os
src = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\exa_search_results.json"
data = json.load(open(src, encoding="utf-8"))
data = [d for d in data if d.get("label") not in {"h", "S", "s"}]
with open(src, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Cleaned: {len(data)} queries remain")

lines = []
lines.append("# Sources -- Anime Super-Resolution Landscape 2026")
lines.append("")
lines.append("**Generated:** 2026-09-03  ")
lines.append("**Total queries:** " + str(len(data)) + " Exa /search calls  ")
lines.append("**Total hits:** " + str(sum(len(d.get("results", [])) for d in data)) + " URLs with highlights  ")
lines.append("**Method:** exa-py 2.20.0 Python SDK, num_results=8 per query, contents.highlights (numSentences=4, highlightsPerUrl=3)  ")
lines.append("**Total cost:** ~$0.13  ")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Per-Query Index")
lines.append("")
for i, d in enumerate(data, 1):
    lines.append("### " + str(i) + ". " + d.get("label", "?"))
    lines.append("")
    lines.append("**Query:** `" + d.get("query", "") + "`")
    lines.append("")
    lines.append("| # | Title | URL | Date |")
    lines.append("|---|---|---|---|")
    for j, r in enumerate(d.get("results", []), 1):
        title = (r.get("title") or "").replace("|", "\\|").replace(chr(10), " ")[:90]
        url = r.get("url", "")
        pub = r.get("published_date", "") or "--"
        lines.append("| " + str(j) + " | " + title + " | " + url + " | " + pub + " |")
    lines.append("")
    lines.append("#### Highlights (first hit only)")
    lines.append("")
    for j, r in enumerate(d.get("results", [])[:3], 1):
        lines.append("**[" + str(j) + "] " + (r.get("title") or "")[:80] + "**")
        lines.append("URL: " + r.get("url", ""))
        for h in (r.get("highlights") or [])[:1]:
            cleaned = h[:500].replace(chr(10), " ").replace(chr(13), " ").strip()
            lines.append("> " + cleaned)
        lines.append("")

dst = r"E:\\python projects\\upscale_anime\\docs\\research\\anime_sr_2026\\SOURCES.md"
with open(dst, "w", encoding="utf-8") as f:
    f.write(chr(10).join(lines))
print("SOURCES.md written:", os.path.getsize(dst), "bytes,", len(lines), "lines")