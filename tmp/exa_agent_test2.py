
import exa_py, json
e = exa_py.Exa(api_key="0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")
run = e.agent.runs.create_and_wait(
    query="List 3 AI infrastructure companies hiring founding designers",
    output_schema={"type":"object","properties":{"companies":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"url":{"type":"string"}}}}}, "required":["companies"]},
    timeout_ms=90000,
)
print("STATUS:", run.status)
print("OUTPUT:", json.dumps(run.output.model_dump() if run.output else None, indent=2)[:2000])
print("COST:", getattr(run, "cost_dollars", None))
print("RUN_ID:", getattr(run, "id", None))
print("RUN_ATTRS:", [a for a in dir(run) if not a.startswith("_")])
