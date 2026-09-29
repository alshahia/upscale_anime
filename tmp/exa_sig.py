import exa_py, json
exa = exa_py.Exa("0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")
r = exa.search(
    "Real-ESRGAN animevideov3 model 4x super resolution parameters paper",
    num_results=5,
    type="auto",
    contents={"highlights": {"numSentences": 3, "highlightsPerUrl": 2}},
)
print("resolved_search_type:", r.resolved_search_type)
print("search_time:", r.search_time)
print("cost_dollars:", r.cost_dollars)
print("num results:", len(r.results))
for i, x in enumerate(r.results):
    print(f"\n[{i}] {x.title[:90]}")
    print(f"    url: {x.url}")
    print(f"    publishedDate: {x.publishedDate}")
    print(f"    author: {x.author}")
    if x.highlights:
        for h in x.highlights[:2]:
            print(f"    hl: {h[:200]}")
