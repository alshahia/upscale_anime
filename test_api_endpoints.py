import io
import requests
from PIL import Image
import numpy as np
import concurrent.futures

BASE = "http://127.0.0.1:8000"

img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
buf = io.BytesIO()
img.save(buf, format="PNG")
img_bytes = buf.getvalue()

def post_inference(scale_factor, model_name="checkpoint_compatible"):
    files = {"image": ("test.png", img_bytes, "image/png")}
    data = {"model_name": model_name, "scale_factor": scale_factor}
    return requests.post(BASE + "/inference", data=data, files=files, timeout=120)

results = {}

print("=== Test 1: POST /inference valid (scale=4) ===")
r = post_inference(4)
print("status:", r.status_code)
if r.status_code == 200:
    j = r.json()
    print("success:", j.get("success"))
    print("input_size:", j.get("image_size"))
    print("output_size:", j.get("output_size"))
    print("scale_factor:", j.get("scale_factor"))
    print("message:", j.get("message"))
    results["valid_200"] = True
    results["upscaled"] = (j.get("output_size", {}).get("width") == 256 and j.get("output_size", {}).get("height") == 256)
else:
    print("body:", r.text[:500])
    results["valid_200"] = False
    results["upscaled"] = False

print()
print("=== Test 2: POST /inference invalid scale (scale=5) ===")
r = post_inference(5)
print("status:", r.status_code)
print("body:", r.text[:300])
results["invalid_400"] = r.status_code == 400

print()
print("=== Test 3: GET /inference/checkpoint_compatible (stub removed) ===")
r = requests.get(BASE + "/inference/checkpoint_compatible", params={"image_url": "http://x/y.png"}, timeout=30)
print("status:", r.status_code)
print("body:", r.text[:300])
results["stub_removed"] = r.status_code == 404

print()
print("=== Test 4: GET /health ===")
r = requests.get(BASE + "/health", timeout=30)
print("status:", r.status_code)
if r.status_code == 200:
    j = r.json()
    print("status:", j.get("status"), "| models_loaded:", j.get("models_loaded"))

print()
print("=== Test 5: concurrent requests (model count stays 1) ===")
def make_request(i):
    return post_inference(4).status_code
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    codes = list(ex.map(make_request, range(4)))
print("concurrent status codes:", codes)
r = requests.get(BASE + "/health", timeout=30)
j = r.json()
print("models_loaded after concurrent:", j.get("models_loaded"))
results["no_double_load"] = j.get("models_loaded") == 1

print()
print("=== Test 6: invalid scale does NOT load a model (model count unchanged) ===")
# Unload the model first is not possible via API easily; instead verify 400 returns fast
# and that a bad scale on a fresh model name still 400s. Here we just confirm 400 path.
r = post_inference(7)
print("status for scale=7:", r.status_code)
results["invalid_400_7"] = r.status_code == 400

print()
print("=== SUMMARY ===")
for k, v in results.items():
    print("  " + k + ": " + ("PASS" if v else "FAIL"))
print("ALL PASS:", all(results.values()))
