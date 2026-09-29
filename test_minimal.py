from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import io
import json
import httpx

app = FastAPI()

class Req(BaseModel):
    name: str
    scale: int = 4

@app.post("/test")
async def test_ep(request: Req, image: UploadFile = File(...)):
    return {"name": request.name, "scale": request.scale, "img": image.filename}

# Test with TestClient (httpx-based)
from fastapi.testclient import TestClient
client = TestClient(app)

print("=== TestClient: request as JSON string ===")
r = client.post("/test", data={"request": json.dumps({"name": "foo", "scale": 4})}, files={"image": ("t.png", b"x", "image/png")})
print("status:", r.status_code)
print("body:", r.text[:300])

print()
print("=== TestClient: request as dict via json field ===")
# Try sending as a JSON body with file
r = client.post("/test", data={"request": json.dumps({"name": "foo", "scale": 4})}, files={"image": ("t.png", b"x", "image/png")})
print("status:", r.status_code)
print("body:", r.text[:300])
