# Drift Detection Workflow

**Purpose**: Detect and prevent deviations from Hybrid Agent System standards.

**Tool**: `scripts/drift_detection.py`

---

## What is Drift?

Drift occurs when code gradually deviates from established standards:

```python
# Standard (HYBRID_AGENT_SYSTEM.md)
def process(data: str | None) -> Path:
    return Path(data) if data else None

# Drift (old patterns creeping back in)
from typing import Optional  # ← Drift: importing deprecated typing
import os.path              # ← Drift: using os.path instead of pathlib
def process(data: Optional[str]) -> str:  # ← Drift: Optional instead of |
    return os.path.join("a", data)        # ← Drift: os.path usage
```

---

## Detection Rules

| Code | Severity | Rule | Fix |
|------|----------|------|-----|
| `pip install` | **ERROR** | D001 | Use `uv add` |
| `Optional[X]` | **ERROR** | D002 | Use `X \| None` |
| `Union[X,Y]` | **ERROR** | D003 | Use `X \| Y` |
| `os.path.*` | **ERROR** | D004 | Use `pathlib.Path` |
| `print()` | WARNING | D005 | Use `logger.info()` |
| bare `except:` | **ERROR** | D006 | Catch specific exceptions |
| `typing.Any` | **ERROR** | D007 | Use specific types |
| `# TODO/FIXME` | INFO | D008 | Move to TASK.md |
| `open()` w/o encoding | WARNING | D009 | Use `encoding="utf-8"` |
| f-strings in logs | INFO | D010 | Use structured logging |
| file > 400 lines | **ERROR** | SIZE001 | Split into modules |
| function > 50 lines | WARNING | FUNC001 | Extract helpers |

---

## Usage

### Local Development

```bash
# Quick check
python scripts/drift_detection.py

# Check specific directory
python scripts/drift_detection.py src/agent/

# JSON output for scripts
python scripts/drift_detection.py --json

# Auto-fix where possible (D002, D003)
python scripts/drift_detection.py --fix

# Treat warnings as errors (CI mode)
python scripts/drift_detection.py --fail-on-warning
```

### Pre-Commit Hook

Add to `.git/hooks/pre-commit` or use [pre-commit](https://pre-commit.com/):

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: drift-detection
        name: Hybrid Agent System Drift Check
        entry: python scripts/drift_detection.py --fail-on-warning
        language: system
        pass_filenames: false
        always_run: true
```

### CI Integration

```yaml
# .github/workflows/ci.yml (add this job)
drift-detection:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v3
    - run: uv run python scripts/drift_detection.py --json
      id: drift
    - if: failure()
      run: |
        echo "::error::Hybrid Agent System drift detected"
        echo "${{ steps.drift.outputs }}"
```

---

## Example Output

### Text Format (Default)

```
======================================================================
DRIFT DETECTION REPORT
======================================================================
Files scanned: 47
Lines scanned: 12,847
Violations: 3 (2 errors, 1 warning)

======================================================================
❌ ERRORS (blocking)
======================================================================

src/whatsapp/client.py:42
  [D002] Found 'Optional[X]' - use 'X | None' instead
  💡 Replace 'Optional[str]' with 'str | None'
  🔧 Auto-fix available: --fix

src/api/routes.py:128
  [SIZE001] File has 487 lines (max 400)
  💡 Split into multiple files or extract utilities

======================================================================
⚠️  WARNINGS (non-blocking)
======================================================================

src/utils/helpers.py:15
  [D005] Found 'print()' - use structured logging instead
  💡 Replace 'print(msg)' with 'logger.info(msg, extra={...})'
```

### JSON Format

```json
{
  "files_scanned": 47,
  "lines_scanned": 12847,
  "error_count": 2,
  "warning_count": 1,
  "violations": [
    {
      "file": "src/whatsapp/client.py",
      "line": 42,
      "rule": "D002",
      "severity": "ERROR",
      "message": "Found 'Optional[X]' - use 'X | None' instead",
      "suggestion": "Replace 'Optional[str]' with 'str | None'",
      "auto_fixable": true
    }
  ]
}
```

---

## Auto-Fix Capabilities

Some rules support automatic correction:

| Rule | Auto-Fix | Example |
|------|----------|---------|
| D002 | ✅ | `Optional[str]` → `str \| None` |
| D003 | ✅ | `Union[str, int]` → `str \| int` |

**Manual review required** for:
- Import changes (D004 - need to add pathlib import)
- Logic changes (D006 - need to choose correct exception type)
- Architecture changes (SIZE001, FUNC001 - need to design split)

---

## Drift Prevention Strategies

### 1. IDE Integration

Configure your editor to highlight drift patterns:

```json
// .vscode/settings.json
{
  "python.analysis.diagnosticSeverityOverrides": {
    "reportDeprecated": "error"
  },
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  }
}
```

### 2. Code Review Checklist

Reviewers should verify:
- [ ] No `Optional[X]` or `Union[X,Y]` in new code
- [ ] No `os.path` usage
- [ ] All functions have type annotations
- [ ] No files approaching 400 lines
- [ ] `print()` calls are justified (debug only)

### 3. Weekly Drift Audit

```bash
# Add to cron or CI weekly job
python scripts/drift_detection.py --json > drift-report.json
# Track trend: error_count should be 0 and decreasing
```

---

## Common Drift Scenarios

### Scenario 1: Copy-Paste from Stack Overflow

```python
# Copied code (drift)
import os
path = os.path.join(BASE_DIR, "config.yaml")

# After drift detection
from pathlib import Path
path = Path(BASE_DIR) / "config.yaml"
```

### Scenario 2: Habitual Typing

```python
# Habit (drift)
from typing import Optional
def find(id: Optional[int]) -> Optional[User]:
    ...

# Correct
from __future__ import annotations
def find(id: int | None) -> User | None:
    ...
```

### Scenario 3: Quick Debugging

```python
# Temporary (drift)
print(f"DEBUG: user_id={user_id}")

# Correct
logger.debug("Processing user", extra={"user_id": user_id})
```

---

## Adding New Detection Rules

To extend drift detection, edit `scripts/drift_detection.py`:

```python
DRIFT_RULES: dict[str, dict] = {
    # ... existing rules ...
    
    "D011": {
        "pattern": r"requests\.get\(",
        "severity": "WARNING",
        "message": "Found requests usage - prefer httpx for async support",
        "suggestion": "Replace requests with httpx and use async where possible",
        "auto_fixable": False,
    },
}
```

Guidelines for new rules:
1. **Severity**: ERROR only for standards violations, WARNING for recommendations
2. **Pattern**: Use specific regex that minimizes false positives
3. **Suggestion**: Provide actionable fix with example
4. **Auto-fixable**: Only if safe to regex-replace without breaking logic

---

## Troubleshooting

### False Positives

If a rule incorrectly flags valid code:

```python
# Legitimate use of pattern
data = {"key": "value"}  # D010 might flag this as f-string related

# Suppression (use sparingly)
# drift:ignore=D010  # JSON data, not logging
```

**Note**: Add suppression comments only when necessary and document why.

### Performance

For large codebases (>10k files):

```bash
# Scan only changed files
python scripts/drift_detection.py $(git diff --name-only HEAD~1)

# Parallel scan (if implemented)
python scripts/drift_detection.py --jobs=4
```

---

## Related Documents

- `HYBRID_AGENT_SYSTEM.md` — Standards that drift detection enforces
- `AGENT_ONBOARDING.md` — New agent guide (includes drift prevention)
- `ci-validation.md` — CI quality gate documentation
- `scripts/drift_detection.py` — Implementation

---

*Document maintained as part of TASK-026: Prompt System Drift Detection*
