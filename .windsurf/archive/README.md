# Archived Workflow Files

**Archive Date**: 2024-04-21  
**Superseded By**: `.windsurf/HYBRID_AGENT_SYSTEM.md`  
**Reason**: Consolidation of 7+ prompt systems into unified architecture

---

## Archive Structure

```
.windsurf/archive/
├── README.md              # This file
├── rules/                 # Archived rule files from .windsurf/rules/
│   ├── UNIVERSAL_PYTHON_AGENT_BASE_PROMPT.md
│   ├── FRONTEND DEVELOPMENT AGENT — SYSTEM PROMPT v2.md
│   ├── ENHANCED AGENT OVERLAY PROMPT — v1.0.md
│   ├── SYSTEM PROMPT — Python Base Agent.md
│   └── SYSTEM PROMPT — Python Polymath Engineer.md
└── workflows/             # Archived workflow files from .windsurf/workflows/
    ├── (various workflow files)
```

---

## Superseded Files

### Rule Files (now in `archive/rules/`)

| Original File | Strengths Preserved in Hybrid |
|---------------|------------------------------|
| `UNIVERSAL_PYTHON_AGENT_BASE_PROMPT.md` | Python 3.10+ enforcement, uv package manager, type safety, exception hierarchy |
| `FRONTEND DEVELOPMENT AGENT — SYSTEM PROMPT v2.md` | WCAG 2.1 AA, OWASP Top 10, component patterns, i18n |
| `ENHANCED AGENT OVERLAY PROMPT — v1.0.md` | Multi-agent coordination, self-audit protocol, CI/CD awareness |
| `SYSTEM PROMPT — Python Base Agent.md` | Clean structure, logging, configuration management |
| `SYSTEM PROMPT — Python Polymath Engineer.md` | Domain-specific patterns (Data/AI/Web), vectorization |

### Workflow Files (now in `archive/workflows/`)

All workflow files from `.windsurf/workflows/` have been archived. The unified system is now documented in:

- **Single Source of Truth**: `.windsurf/HYBRID_AGENT_SYSTEM.md`
- **Task Tracking**: `TASK.md` (Phase 6)
- **Project Planning**: `PLANNING.md` (Agent Workflow Architecture section)

---

## How to Use the Archive

1. **Reference Only**: These files are for historical reference
2. **Do Not Modify**: Changes should go to `HYBRID_AGENT_SYSTEM.md`
3. **New Agents**: Should read `HYBRID_AGENT_SYSTEM.md`, not these archived files
4. **Research**: If investigating why a specific standard exists, check the source file here

---

## Migration Guide

### For Existing Agents

If you were trained on one of these files, note these key changes:

| Old Approach | New Hybrid Approach |
|-------------|---------------------|
| `Optional[X]` | `X \| None` (Python 3.10+) |
| `Union[X, Y]` | `X \| Y` |
| `pip install` | `uv add` only |
| `os.path.join()` | `Path() /` |
| `print()` | `logging.getLogger(__name__).info()` |
| File size unchecked | Max 400 lines (refactor at 350) |
| No self-audit | Mandatory pre-submission checklist |

---

## Related Documents

- **Active System**: `.windsurf/HYBRID_AGENT_SYSTEM.md`
- **Project Tasks**: `TASK.md` (TASK-022 through TASK-025)
- **Project Plan**: `PLANNING.md` (Agent Workflow Architecture section)
- **Onboarding**: `.windsurf/AGENT_ONBOARDING.md` (when created)

---

*Archive maintained as part of TASK-023: Workflow File Consolidation*
