
import exa_py, inspect
e = exa_py.Exa(api_key="0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")
print("CREATE sig:", inspect.signature(e.agent.runs.create))
print("CREATE_AND_WAIT sig:", inspect.signature(e.agent.runs.create_and_wait))
print("POLL sig:", inspect.signature(e.agent.runs.poll_until_finished))
