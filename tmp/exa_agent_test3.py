
import exa_py, json, time
e = exa_py.Exa(api_key="0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")
t0 = time.time()
try:
    run = e.agent.runs.create_and_wait(
        query="Find 3 AI startups with founding designer openings",
        output_schema={"type":"object","properties":{"companies":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"url":{"type":"string"}}}}}, "required":["companies"]},
        timeout_ms=300000,
    )
    print(f"ELAPSED: {time.time()-t0:.1f}s")
    print("STATUS:", run.status)
    print("OUTPUT:", json.dumps(run.output.model_dump() if run.output else None, indent=2)[:2000])
    print("COST_DOLLARS:", getattr(run, "cost_dollars", None))
    print("RUN_ID:", getattr(run, "id", None))
except Exception as exc:
    print(f"ELAPSED: {time.time()-t0:.1f}s")
    print(f"EXC TYPE: {type(exc).__name__}")
    print(f"EXC: {exc}")
