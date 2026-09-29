"""Reusable Exa agent.runs helper for structured research."""
import os
import json
import time
from typing import Any, Optional
import exa_py


def _safe(obj):
    """Best-effort JSON serializer for pydantic / dataclass / unknown types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return _safe(obj.model_dump())
    if hasattr(obj, "dict"):
        return _safe(obj.dict())
    return str(obj)


def exa_research(
    query: str,
    schema: dict,
    system_prompt: Optional[str] = None,
    timeout_ms: int = 300000,
    previous_run_id: Optional[str] = None,
) -> dict:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY not set in environment")
    e = exa_py.Exa(api_key=api_key)
    kwargs = dict(query=query, output_schema=schema, timeout_ms=timeout_ms)
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    if previous_run_id:
        kwargs["previous_run_id"] = previous_run_id
    t0 = time.time()
    run = e.agent.runs.create_and_wait(**kwargs)
    elapsed = time.time() - t0
    return _safe({
        "run_id": getattr(run, "id", None),
        "status": str(getattr(run, "status", "?")),
        "elapsed_s": round(elapsed, 2),
        "output": run.output.model_dump() if run.output else None,
        "cost_dollars": getattr(run, "cost_dollars", None),
    })


if __name__ == "__main__":
    r = exa_research(
        query="List 1 open-source anime super-resolution project on GitHub updated 2024-2026",
        schema={
            "type": "object",
            "properties": {
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "url": {"type": "string"},
                            "license": {"type": "string"},
                            "year": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["projects"],
        },
        system_prompt="Prefer projects updated 2024-2026. Include GitHub URL and license.",
    )
    print(json.dumps(r, indent=2))
