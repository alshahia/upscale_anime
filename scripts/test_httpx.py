import io
import json
import httpx
from PIL import Image
import numpy as np

BASE = "http://127.0.0.1:8000"
img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

req_data = {"model_name": "checkpoint_compatible", "scale_factor": 4, "enhance_faces": False, "tile_size": None, "return_base64": False}

# httpx: request as JSON string in multipart
print("=== httpx: request as JSON string ===")
with httpx.Client(timeout=120) as client:
    files = {"image": ("test.png", img_bytes, "image/png")}
    data = {"request": json.dumps(req_data)}
    r = client.post(BASE + "/inference", data=data, files=files)
    print("status:", r.status_code)
    print("body:", r.text[:400])

# httpx: request as JSON string with content-type
print()
print("=== httpx: request as JSON string (content-type json) ===")
with httpx.Client(timeout=120) as client:
    files = {
        "image": ("test.png", img_bytes, "image/png"),
        "request": (None, json.dumps(req_data), "application/json"),
    }
    r = client.post(BASE + "/inference", files=files)
    print("status:", r.status_code)
    print("body:", r.text[:400])
