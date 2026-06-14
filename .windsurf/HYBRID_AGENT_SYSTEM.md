# HYBRID AGENT SYSTEM — Unified Workflow Architecture

**Version**: 1.0.0 | **Status**: Active | **Scope**: All Agent Interactions

---

## 🎯 Hybrid Solution Overview

This document consolidates **7+ agent prompt systems** into a single, cohesive architecture:

| Source System | Strengths | Hybrid Integration |
|---------------|-----------|-------------------|
| UNIVERSAL PYTHON AGENT | Strong Python standards, uv enforcement, type safety | Core Python execution layer |
| FRONTEND DEVELOPMENT AGENT | WCAG/OWASP enforcement, component patterns | UI/UX task handling |
| ENHANCED OVERLAY | Multi-agent coordination, CI/CD awareness | Orchestration layer |
| PYTHON POLYMATH | Data/AI/Web domain expertise | Domain-specific reasoning |
| SYSTEM PROMPT — Python Base | Clean structure, logging, config patterns | Foundation patterns |

---

## 🏗️ Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HYBRID AGENT SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER (from Enhanced Overlay)                          │
│  ├── Multi-agent handoff protocol                                       │
│  ├── Task state management (TASK.md integration)                      │
│  ├── CI/CD pipeline awareness                                           │
│  └── Security escalation ladder                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  EXECUTION LAYER (from Universal Python + Python Base)                │
│  ├── Python 3.10+ enforcement (| instead of Union)                    │
│  ├── uv package manager (strict - no pip fallback)                    │
│  ├── Type hints mandatory (no Any, no Optional[X])                   │
│  ├── Custom exception hierarchy (AppError base)                        │
│  ├── Pydantic at all IO boundaries                                     │
│  └── Structured logging (no print())                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  DOMAIN LAYER (from Python Polymath)                                  │
│  ├── Data: Vectorized pandas/polars (no row loops)                      │
│  ├── AI/ML: PyTorch patterns + device-agnostic code                   │
│  ├── Web: FastAPI service layer pattern                               │
│  └── CLI: Typer with pathlib                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  FRONTEND LAYER (from Frontend Agent)                                 │
│  ├── WCAG 2.1 AA accessibility (non-negotiable)                         │
│  ├── OWASP Top 10 controls                                            │
│  ├── Component co-location (test + source together)                   │
│  └── i18n-ready string handling                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  QUALITY LAYER (consolidated from all)                                │
│  ├── Self-audit mandatory before output                               │
│  ├── File limit: 400 lines (refactor at 350)                          │
│  ├── Function limit: 50 lines                                         │
│  └── Quality gate: ruff → mypy → bandit → pytest                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Hybrid Standards (Best of All Systems)

### 1. Tool Manifest (Unified)

```
ACTIVE TOOLS:
┌─────────────────────┬──────────────────────────┬─────────────┐
│ Tool                │ Capability               │ Status      │
├─────────────────────┼──────────────────────────┼─────────────┤
│ read_file(path)     │ Read with line ranges   │ [✓] enabled │
│ write_file(p, c)    │ Atomic write            │ [✓] enabled │
│ list_dir(path)      │ Recursive listing       │ [✓] enabled │
│ run_shell(cmd)      │ Execute with cwd        │ [✓] enabled │
│ run_tests(p)        │ pytest with coverage    │ [✓] enabled │
│ git_status()        │ Working tree state       │ [✓] enabled │
│ grep_search(q, p)   │ ripgrep search          │ [✓] enabled │
│ code_search(q, p)   │ Parallel search agent    │ [✓] enabled │
│ todo_list(t)        │ Task management          │ [✓] enabled │
│ browser_preview(u)  │ Live preview             │ [✓] enabled │
└─────────────────────┴──────────────────────────┴─────────────┘

**CROSS-PLATFORM COMMAND EXECUTION:**
```powershell
# Windows PowerShell (use semicolons, not && or ||)
run_command("python script.py; echo 'done'")  # ✓ Works
run_command("python script.py && echo 'done'")  # ✗ Fails

# Bash/Linux (use && and ||)
run_command("python script.py && echo 'done'")  # ✓ Works

# Syntax-only Python verification (no import needed)
python -m py_compile script.py  # ✓ Validates syntax without executing
```

MODE DETECTION:
- MODE B (Tool-Enabled): All tools active → execute directly
- MODE C (Hybrid): Partial tools → use available, narrate unavailable
- MODE A (Chat): No tools → code blocks + manual commands
```

### 2. Python Standards (Strict - from Universal Python)

```python
# ✓ Modern syntax (Python 3.10+)
def process(value: str | None) -> dict[str, int]: ...  # NOT Optional[str]

# ✓ Pathlib (never os.path)
from pathlib import Path
config_path = Path("config") / "settings.json"

# ✓ Structured logging (never print)
logger = logging.getLogger(__name__)
logger.info("event", extra={"key": value})

# ✓ Exception hierarchy
class AppError(Exception): ...
class ValidationError(AppError): ...

# ✗ Banned patterns
Any  # Use object, Protocol, or generics
Optional[X]  # Use X | None
Union[X, Y]  # Use X | Y
Dict/List  # Use dict/list
print()  # Use logging
os.path.join()  # Use Path
```

### 3. Self-Audit Protocol (Mandatory)

```
PRE-SUBMISSION CHECKLIST — RUN SILENTLY:

CORRECTNESS     [ ] Logic matches requirement
                [ ] Edge cases handled (None, empty, overflow)
                [ ] No bare except — always specific types

TYPE SAFETY     [ ] All functions fully annotated
                [ ] No Any, no Optional, no Union
                [ ] Pydantic at IO boundaries

SECURITY        [ ] No hardcoded secrets
                [ ] shell=False only
                [ ] No eval/exec/pickle on untrusted data

CODE QUALITY    [ ] File < 400 lines
                [ ] Function < 50 lines
                [ ] Google-style docstrings
                [ ] "Reason:" comments for non-obvious

TESTING         [ ] New code has tests
                [ ] Happy path + error cases
                [ ] pytest passes

OUTPUT:
[AUDIT: N/N ✓ | Fixed: X | Deferred: Y (see TASK.md)]
```

### 4. Multi-Agent Handoff (Enhanced Protocol)

```markdown
## Handoff Schema (JSON)
{
  "agent": "python-agent",
  "task_id": "TASK-XXX",
  "status": "completed|blocked|failed",
  "summary": "One sentence",
  "files_changed": ["path: description"],
  "quality": {
    "pytest": "N/N",
    "mypy": "clean|N errors",
    "ruff": "clean|N violations",
    "bandit": "clean|issues",
    "coverage": "N%"
  },
  "assumptions": ["[ASSUMED] item"],
  "blockers": ["if any"],
  "next": "Next action"
}
```

### 5. Domain-Specific Patterns

#### Data Analysis (Vectorized)
```python
# ✓ Method chaining, no loops
def clean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(ts=lambda x: pd.to_datetime(x['ts']))
        .loc[lambda x: x['status'] == 'active']
        .groupby('region')
        .agg(mean=('value', 'mean'))
        .reset_index()
    )
# Scale > 10GB RAM → suggest polars/dask
```

#### AI/ML (Device-Agnostic)
```python
# ✓ Explicit device handling
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)

# ✓ DataLoader best practices
DataLoader(dataset, num_workers=4, pin_memory=True)
```

#### Web API (Service Layer)
```python
# ✓ Business logic in services, not views
@router.post("/users")
async def create(
    payload: CreateUserRequest,
    service: UserService = Depends()
) -> UserResponse:
    user = await service.create(payload)
    return UserResponse.model_validate(user)
```

#### Frontend (WCAG/OWASP)
```typescript
// ✓ Accessibility first
<button aria-label="Close dialog" onClick={handleClose}>
  <XIcon />
</button>

// ✓ Security - no innerHTML without DOMPurify
<div dangerouslySetInnerHTML={{__html: DOMPurify.sanitize(html)}} />
```

#### Python Import Resolution
```python
# ✓ Use Absolute Imports Always (never relative)
# In __init__.py files:
from package.module import Class  # ✓ Correct
# from .module import Class       # ✗ Wrong - fails when scripts run directly

# ✓ Script execution pattern at project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from package.module import Thing  # ✓ Works in all contexts

# ✓ Package structure with proper exports
# src/__init__.py:
from src.module import ClassA, ClassB
__all__ = ['ClassA', 'ClassB']
```

#### Defensive Data Validation
```python
# ✓ Always validate early with helpful error messages
def validate_dataset(path: Path) -> dict:
    """Validate dataset exists and contains valid data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Please create the directory or update the config.\n"
            f"To create test data: python scripts/create_test_data.py --output {path}"
        )
    
    files = list(path.glob("*.png")) + list(path.glob("*.jpg"))
    if len(files) == 0:
        raise ValueError(
            f"No images found in {path}\n"
            f"Supported formats: .png, .jpg, .jpeg\n"
            f"Add images to this directory or update the data path in config."
        )
    
    return {'valid': True, 'count': len(files)}

# ✓ Validate at initialization, not during first batch
# ✓ Error messages should: state problem, suggest fix, provide example command
```

#### Optional Dependency Fallback
```python
# ✓ Handle optional ML dependencies gracefully
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

class FallbackMamba(nn.Module):
    """Pure-Python fallback maintaining identical API."""
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        # Implement with standard PyTorch ops
        self.layers = nn.ModuleList([...])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Functional equivalent, possibly slower
        for layer in self.layers:
            x = layer(x)
        return x

def create_mamba(d_model: int, **kwargs) -> nn.Module:
    """Factory returning optimal or fallback implementation."""
    if MAMBA_AVAILABLE:
        return Mamba(d_model, **kwargs)
    else:
        print("Warning: Using fallback Mamba. Install mamba-ssm for better performance.")
        return FallbackMamba(d_model, **kwargs)

# ✓ Key principles:
# - Fallback maintains identical API (duck typing)
# - Warn user about suboptimal performance
# - Don't fail — degrade gracefully
# - Document optional dependency clearly
```

---

## 📋 Task Integration

### TASK.md Updates Required

```markdown
### TASK-022: Hybrid Agent System Implementation
**Status**: in_progress
**Priority**: high
**Effort**: M

**Goal**: Consolidate all agent workflow files into unified Hybrid Agent System.

**Acceptance Criteria**:
- [ ] HYBRID_AGENT_SYSTEM.md created with consolidated standards
- [ ] All 7+ source prompt files reviewed and integrated
- [ ] Self-audit protocol mandatory
- [ ] Multi-agent handoff schema defined
- [ ] Domain-specific patterns documented
- [ ] CI/CD pipeline alignment specified

**Files Created**:
- `.windsurf/HYBRID_AGENT_SYSTEM.md` - This unified document
- `.windsurf/.agentrc` - Agent configuration (optional)

**Files Modified**:
- `PLANNING.md` - Add Agent Architecture section
- `TASK.md` - Add TASK-022 and subtasks
```

---

## � CI/CD Quality Gate

The CI/CD pipeline enforces hybrid standards automatically. All code must pass these checks in order:

### Quality Gate Sequence

```bash
# 1. FORMAT (fastest)
uv run ruff format --check .
# Fail if: Any file needs formatting
# Fix: uv run ruff format .

# 2. LINT
uv run ruff check . --fix
# Fail if: Style violations, unused imports
# Fix: Auto-fixes applied, manual review for remaining

# 3. SECURITY
uv run bandit -r config/ whatsapp/ llm/ storage/ agent/ api/ utils/ -ll
# Fail if: Hardcoded secrets, shell=True, eval/exec, SQL injection risks
# Fix: Remove secrets, use shell=False, parameterized queries

# 4. TYPE CHECK
uv run mypy config/ whatsapp/ llm/ storage/ agent/ api/ utils/
# Fail if: Missing annotations, Any types, Optional[X] instead of X | None
# Fix: Add annotations, replace Any with specific types

# 5. FILE SIZE CHECK�find . -name "*.py" -not -path "./.venv/*" -exec wc -l {} + | awk '$1 > 400 {exit 1}'
# Fail if: Any file exceeds 400 lines
# Fix: Refactor into multiple files

# 6. TESTS (slowest)
uv run pytest --cov=. --cov-report=xml --cov-report=term --cov-fail-under=80
# Fail if: Any test fails or coverage < 80%
# Fix: Add tests for new code, ensure happy + error paths covered
```

### CI Configuration

See `.github/workflows/ci.yml` for the actual implementation. Key features:
- Uses `uv` for dependency management (not pip)
- Runs on every push to main/develop and all PRs
- PostgreSQL service container for integration tests
- Fails fast (stops at first error)
- Docker build verification as final step

### Local Pre-Commit

Run the quality gate before committing:

```bash
# Quick check (format + lint + security)
uv run ruff format --check . && uv run ruff check . && uv run bandit -r src/ -ll

# Full check (adds type check + tests)
uv run ruff format --check . && uv run ruff check . && uv run bandit -r src/ -ll && uv run mypy src/ && uv run pytest
```

### Drift Detection

The `scripts/drift_detection.py` tool scans for patterns that violate hybrid standards:
- `pip install` → should be `uv add`
- `Optional[X]` → should be `X | None`
- `os.path` → should be `pathlib.Path`
- `print()` → should use `logger.info()`
- Files > 400 lines → should be split

Run locally:
```bash
python scripts/drift_detection.py          # Text output
python scripts/drift_detection.py --json # JSON for CI
python scripts/drift_detection.py --fix  # Auto-fix where possible
```

---

## �🔄 Continuous Updates

### Weekly Review Tasks
1. **Audit agent outputs** against hybrid standards
2. **Update TASK.md** with new discoveries
3. **Refactor files approaching 400 lines**
4. **Review quality gate results**

### Monthly Enhancement
1. **Review new domain patterns** from community
2. **Update Python version** baseline (currently 3.10+)
3. **Refresh security rules** (OWASP updates)
4. **Archive superseded patterns** to `.windsurf/archive/`

---

## 🎓 Training Mode

For new agents joining the project:

```bash
# 1. Read core documents
read_file("PLANNING.md")
read_file("TASK.md")
read_file(".windsurf/HYBRID_AGENT_SYSTEM.md")

# 2. Verify environment
uv sync  # Install dependencies
uv run pytest  # Verify tests pass

# 3. Check current task
todo_list({"todos": [{"id": "1", "content": "Read hybrid system docs", "status": "completed", "priority": "high"}]})
```

---

*End of Hybrid Agent System v1.0*
