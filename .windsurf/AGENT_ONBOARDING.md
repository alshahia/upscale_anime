# Agent Onboarding Guide

**Version**: 1.0 | **Project**: Social Agent WhatsApp AI | **Date**: 2024-04-21

---

## 🚀 Welcome, Agent!

This guide will get you up to speed with the project in **3 simple steps**. Read this carefully before starting any work.

---

## Step 1: Read the Core Documents (REQUIRED)

Read these **in order**:

| Order | Document | Purpose | Time |
|-------|----------|---------|------|
| 1 | `PLANNING.md` | Project vision, stack, architecture, constraints | 5 min |
| 2 | `TASK.md` | Active tasks, backlog, completed work | 3 min |
| 3 | `.windsurf/HYBRID_AGENT_SYSTEM.md` | **Your operating manual** — how to work on this project | 10 min |

> ⚠️ **CRITICAL**: Never skip Step 3. The Hybrid Agent System defines how you write code, run tests, and self-audit.

---

## Step 2: Verify Your Environment

Run these commands to ensure everything works:

```bash
# 1. Install dependencies (uv only — no pip!)
uv sync

# 2. Run tests (should all pass)
uv run pytest

# 3. Run quality gate (should pass)
uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run bandit -r src/ -ll
```

**Expected Results**:
- `uv sync`: Completes without errors
- `pytest`: 100% pass rate (this project has comprehensive tests)
- `ruff`: No violations
- `mypy`: No type errors
- `bandit`: No security issues

---

## Step 3: Check Current Task

Always know what you're working on:

```bash
# Read the active task section
grep -A 5 "## 🔴 ACTIVE" TASK.md
```

**Current Active Task**: Check `TASK.md` for the latest.

---

## 🎯 The Hybrid Agent System: Quick Reference

### Core Rules (Memorize These)

| Rule | What It Means | Example |
|------|---------------|---------|
| **Python 3.10+** | Use `\|` for unions, never `Optional` | `def f(x: str \| None) -> int:` |
| **uv only** | Never use `pip install` | `uv add package` |
| **pathlib** | Never use `os.path` | `Path("a") / "b"` |
| **structured logging** | Never use `print()` | `logger.info("msg", extra={"key": val})` |
| **type hints** | Every function must be annotated | `def foo(a: int) -> str:` |
| **file size** | Max 400 lines | Refactor at 350 lines |
| **function size** | Max 50 lines | Extract helpers if needed |
| **exception hierarchy** | Use custom AppError base | `raise ValidationError(field, reason)` |
| **Pydantic** | At all IO boundaries | `UserResponse.model_validate(data)` |

### Self-Audit Checklist (Run Before Every Output)

```
CORRECTNESS     [ ] Logic matches requirement
                [ ] Edge cases handled
                [ ] No bare except

TYPE SAFETY     [ ] All functions annotated
                [ ] No Any, Optional, Union
                [ ] Pydantic at boundaries

SECURITY        [ ] No hardcoded secrets
                [ ] shell=False only
                [ ] No eval/exec/pickle

CODE QUALITY    [ ] File < 400 lines
                [ ] Function < 50 lines
                [ ] Google docstrings
                [ ] "Reason:" comments

TESTING         [ ] Tests for new code
                [ ] Happy + error paths
                [ ] pytest passes

OUTPUT:
[AUDIT: N/N ✓ | Fixed: X | Deferred: Y (see TASK.md)]
```

### Quality Gate Order

Always run in this sequence:

```bash
uv run ruff format .       # 1. Format
uv run ruff check . --fix  # 2. Lint (auto-fix)
uv run mypy src/            # 3. Type check
uv run bandit -r src/ -ll   # 4. Security scan
uv run pytest               # 5. Tests (slowest last)
```

---

## 📂 Project Structure

```
social_agent/
├── config/              # Configuration management
├── whatsapp/            # WhatsApp providers (mock, maytapi, wpp)
├── llm/                 # LLM providers (openai, ollama)
├── storage/             # PostgreSQL + pgvector storage
├── agent/               # LangGraph workflow + tools
│   ├── tools/           # smolagents tools
│   └── graph.py         # Main agent orchestration
├── api/                 # FastAPI webhook server
├── utils/               # Logging, helpers
├── tests/               # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── conftest.py      # Fixtures
├── .windsurf/           # Agent workflows (that's us!)
│   ├── HYBRID_AGENT_SYSTEM.md  # Your main guide
│   ├── AGENT_ONBOARDING.md     # This file
│   └── archive/         # Old workflow files
├── PLANNING.md          # Project plan
└── TASK.md              # Task tracking
```

---

## ✅ Example First Task Workflow

Let's say your first task is to add a new tool to the agent:

### 1. Understand the Task

```python
# Read the active task
read_file("TASK.md", offset=1, limit=100)

# Identify what needs to be done
# Example: "Add WeatherLookupTool to agent tools"
```

### 2. Research Existing Patterns

```python
# Look at existing tools for the pattern
grep_search("class.*Tool", "agent/tools/")
read_file("agent/tools/message_search.py", limit=50)

# Check the base class
read_file("agent/tools/base.py")  # If exists, or smolagents.Tool
```

### 3. Plan Your Changes

```markdown
# In your head or notes:
Files to Create:
- agent/tools/weather_lookup.py (the tool)
- tests/agent/tools/test_weather_lookup.py (tests)

Files to Modify:
- agent/tools/__init__.py (export the new tool)
- agent/graph.py (register with smolagents)

Estimate: ~150 lines total (well under 400 limit)
```

### 4. Implement with Self-Audit

```python
# FILE: agent/tools/weather_lookup.py
"""Weather lookup tool for agent."""
from typing import ClassVar
from smolagents import Tool


class WeatherLookupTool(Tool):
    """Look up weather for a location.
    
    Args:
        location: City name or coordinates
    
    Returns:
        Weather data as formatted string
    """
    name: ClassVar[str] = "weather_lookup"
    description: ClassVar[str] = "Get current weather for a location"
    inputs: ClassVar[dict] = {
        "location": {
            "type": "string",
            "description": "City name or coordinates"
        }
    }
    output_type: ClassVar[str] = "string"
    
    def forward(self, location: str) -> str:
        """Fetch weather data."""
        # Reason: Using external API with timeout for reliability
        import httpx
        
        try:
            response = httpx.get(
                f"https://api.weather.example/{location}",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return f"Weather in {location}: {data['summary']}"
        except httpx.TimeoutException:
            return "Weather service unavailable (timeout)"
        except httpx.HTTPStatusError as e:
            return f"Weather lookup failed: {e.response.status_code}"
```

### 5. Run Quality Gate

```bash
uv run ruff format agent/tools/weather_lookup.py
uv run ruff check agent/tools/weather_lookup.py --fix
uv run mypy agent/tools/weather_lookup.py
uv run bandit -r agent/tools/weather_lookup.py -ll
uv run pytest tests/agent/tools/test_weather_lookup.py -v
```

### 6. Self-Audit Output

```
[AUDIT: 5/5 ✓ | Fixed: None | Deferred: None]
```

### 7. Update TASK.md

Mark your sub-tasks as complete in `TASK.md`.

---

## ⚠️ Common Pitfalls (Avoid These!)

### ❌ Pitfall 1: Using pip

```bash
# WRONG
pip install requests

# CORRECT
uv add requests
```

### ❌ Pitfall 2: Using Optional

```python
# WRONG
from typing import Optional
def foo(x: Optional[str]) -> Optional[int]: ...

# CORRECT
def foo(x: str | None) -> int | None: ...
```

### ❌ Pitfall 3: Using os.path

```python
# WRONG
import os
path = os.path.join("config", "settings.yaml")

# CORRECT
from pathlib import Path
path = Path("config") / "settings.yaml"
```

### ❌ Pitfall 4: Using print

```python
# WRONG
print(f"Processing {user_id}")

# CORRECT
import logging
logger = logging.getLogger(__name__)
logger.info("Processing user", extra={"user_id": user_id})
```

### ❌ Pitfall 5: Bare except

```python
# WRONG
try:
    do_something()
except:  # Catches KeyboardInterrupt, SystemExit!
    pass

# CORRECT
try:
    do_something()
except ValueError as e:
    logger.error("Invalid value", error=str(e))
    raise
```

### ❌ Pitfall 6: File too long

```
# WRONG: file at 450 lines, keep adding

# CORRECT: Refactor when approaching 350 lines
# Split into: main.py (200 lines) + utils.py (150 lines)
```

---

## 🆘 When You're Stuck

### Escalation Path

| Situation | Action |
|-----------|--------|
| Unclear requirements | Ask 1 specific question, propose 2 interpretations |
| Deprecation warning | Upgrade to modern equivalent, never silent |
| Security concern | Use escalation ladder (Level 1: explain + alternative) |
| File approaching 400 lines | Propose refactor before continuing |
| Test failing | Fix before claiming complete |
| Uncertain about API | Add `⚠️ Unverified:` prefix + link to docs |

### Emergency Contacts (Conceptual)

```
1. HYBRID_AGENT_SYSTEM.md → Full specification
2. TASK.md → What to do right now
3. PLANNING.md → Why we're doing it
4. tests/ → How things should work
```

---

## 📊 Success Metrics

You'll know you're doing it right when:

- ✅ `ruff check .` passes with 0 violations
- ✅ `mypy src/` reports no errors
- ✅ `pytest` passes 100%
- ✅ No file exceeds 400 lines
- ✅ Every function has type hints
- ✅ Every public function has a docstring
- ✅ Self-audit shows 5/5 checks passing
- ✅ TASK.md is updated with progress

---

## 🎓 Learning Path

### Week 1: Foundation
- [ ] Read all 3 core documents (Step 1)
- [ ] Run environment verification (Step 2)
- [ ] Complete a simple task (e.g., fix typo, add test)
- [ ] Practice self-audit on every output

### Week 2: Integration
- [ ] Complete a medium task (new function, small feature)
- [ ] Run full quality gate before submitting
- [ ] Update TASK.md with proper notes

### Week 3: Mastery
- [ ] Complete a complex task (new module, refactor)
- [ ] Proactive refactoring (split files at 350 lines)
- [ ] Help review other agents' work (multi-agent coordination)

---

## 📝 Checklist: Before You Start Working

- [ ] I've read `PLANNING.md` (understand the project)
- [ ] I've read `TASK.md` (know the active task)
- [ ] I've read `HYBRID_AGENT_SYSTEM.md` (know the rules)
- [ ] I've run `uv sync && uv run pytest` (environment works)
- [ ] I know which TASK-ID I'm working on
- [ ] I've checked for existing similar code (pattern research)

---

## 📝 Checklist: Before You Submit Work

- [ ] Self-audit shows 5/5 passing
- [ ] Quality gate passes: ruff → mypy → bandit → pytest
- [ ] No file exceeds 400 lines
- [ ] Every new function has tests
- [ ] Type hints everywhere (no Any)
- [ ] No hardcoded secrets
- [ ] `print()` replaced with logging
- [ ] `os.path` replaced with pathlib
- [ ] `Optional[X]` replaced with `X | None`
- [ ] TASK.md updated with progress

---

**Remember**: The Hybrid Agent System is designed to help you succeed. Following these standards ensures:
- Your code works reliably
- Other agents can understand and build on your work
- The project maintains high quality as it grows

**Welcome to the team! 🎉**

---

*Document maintained as part of TASK-024: Agent Training Documentation*
