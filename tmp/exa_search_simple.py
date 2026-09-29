import os
import exa_py
print("EXA_API_KEY set:", bool(os.environ.get("EXA_API_KEY")))
print("exa_py version:", getattr(exa_py, "__version__", "?"))
exa = exa_py.Exa("0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")
result = exa.search(
  "how to use exa search ",
  num_results = 5,
  type = "auto",
  contents = {
    "highlights": True
  }
)
print("RESULT TYPE:", type(result))
print("TOP-LEVEL KEYS:", list(result.__dict__.keys()) if hasattr(result, "__dict__") else dir(result))
print("NUM RESULTS:", len(result.results) if hasattr(result, "results") else "?")
for i, r in enumerate((result.results or [])[:3]):
    print(f"\n[{i}] TITLE: {getattr(r, 'title', '?')[:100]}")
    print(f"    URL: {getattr(r, 'url', '?')}")
    hls = getattr(r, "highlights", None)
    if hls:
        for h in hls[:2]:
            print(f"    HL: {h[:180]}")
