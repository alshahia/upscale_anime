
import sys, json
sys.path.insert(0, r"E:\python projects\\upscale_anime\\tmp")
from exa_helper import exa_research
r = exa_research(
    query="Find 1 GitHub repo for an open-source real-time anime upscaler (Python/PyTorch) updated 2024-2026",
    schema={
        "type":"object",
        "properties":{
            "projects":{"type":"array","items":{"type":"object","properties":{
                "name":{"type":"string"},"url":{"type":"string"},"license":{"type":"string"},"year":{"type":"string"}
            }}}
        },
        "required":["projects"]
    },
    system_prompt="Prefer projects updated 2024-2026. Include GitHub URL and license.",
    timeout_ms=300000,
)
print(json.dumps(r, indent=2)[:2500])
