
import urllib.request, json
for pkg in ["mamba-ssm", "causal-conv1d"]:
    r = urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=15)
    data = json.load(r)
    latest = data["info"]["version"]
    files = data["releases"][latest]
    print(f"\n=== {pkg} {latest} ALL files (first 30) ===")
    for f in files[:30]:
        print(f"  {f['filename']}")
    print(f"  TOTAL: {len(files)}")
