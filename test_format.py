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

# Format A: "request" as a JSON form field + file
print("=== Format A: request as JSON form field ===")
req_data = {"model_name": "checkpoint_compatible", "scale_factor": 4, "enhance_faces": False, "tile_size": None, "return_base64": False}
files = {"image": ("test.png", img_bytes, "image/png")}
data = {"request": json.dumps(req_data)}
r = requests.post(BASE + "/inference", data=data, files=files, timeout=120)
print("status:", r.status_code)
print("body:", r.text[:400])

# Format B: individual form fields + file (no nesting)
print()
print("=== Format B: individual form fields ===")
files = {"image": ("test.png", img_bytes, "image/png")}
data = {"model_name": "checkpoint_compatible", "scale_factor": 4}
r = requests.post(BASE + "/inference", data=data, files=files, timeout=120)
print("status:", r.status_code)
print("body:", r.text[:400])
