import io
import json
import requests
from PIL import Image
import numpy as np

BASE = "http://127.0.0.1:8000"
img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

# Check OpenAPI schema for the /inference endpoint
print("=== OpenAPI schema for /inference ===")
r = requests.get(BASE + "/openapi.json", timeout=30)
if r.status_code == 200:
    spec = r.json()
    path = spec["paths"].get("/inference", {})
    post = path.get("post", {})
    print("requestBody:", json.dumps(post.get("requestBody", {}), indent=2)[:1500])
else:
    print("openapi status:", r.status_code)

# Format C: "request" field with explicit JSON content type
print()
print("=== Format C: request field with JSON content-type ===")
req_data = {"model_name": "checkpoint_compatible", "scale_factor": 4, "enhance_faces": False, "tile_size": None, "return_base64": False}
files = {
    "image": ("test.png", img_bytes, "image/png"),
    "request": (None, json.dumps(req_data), "application/json"),
}
r = requests.post(BASE + "/inference", files=files, timeout=120)
print("status:", r.status_code)
print("body:", r.text[:400])
