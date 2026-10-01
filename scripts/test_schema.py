import json
import requests

BASE = "http://127.0.0.1:8000"
r = requests.get(BASE + "/openapi.json", timeout=30)
spec = r.json()
schema = spec["components"]["schemas"].get("Body_inference_endpoint_inference_post", {})
print(json.dumps(schema, indent=2))
