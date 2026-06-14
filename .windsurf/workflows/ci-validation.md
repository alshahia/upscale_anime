# CI Validation Workflow

**Purpose**: Define the exact quality gate sequence for CI/CD alignment with Hybrid Agent System standards.

**Reference**: See `HYBRID_AGENT_SYSTEM.md` for full standards documentation.

---

## Quality Gate Sequence

The CI pipeline enforces this exact order:

```
┌─────────────────────────────────────────────────────────────┐
│  QUALITY GATE                                               │
├─────────────────────────────────────────────────────────────┤
│  1. FORMAT  → uv run ruff format --check .                 │
│     └─ Fast check, fails if files need formatting           │
├─────────────────────────────────────────────────────────────┤
│  2. LINT   → uv run ruff check . --fix                    │
│     └─ Style and best practice violations                  │
├─────────────────────────────────────────────────────────────┤
│  3. SECURITY → uv run bandit -r src/ -ll                    │
│     └─ OWASP Top 10, hardcoded secrets, injection risks      │
├─────────────────────────────────────────────────────────────┤
│  4. TYPE CHECK → uv run mypy src/                          │
│     └─ Strict type checking (no Any, full annotations)     │
├─────────────────────────────────────────────────────────────┤
│  5. FILE SIZE → find . -name "*.py" | wc -l check          │
│     └─ Fail if any file > 400 lines                         │
├─────────────────────────────────────────────────────────────┤
│  6. TESTS  → uv run pytest --cov-fail-under=80            │
│     └─ Slowest check last; 80% coverage minimum            │
└─────────────────────────────────────────────────────────────┘
```

---

## CI/CD Configuration

### GitHub Actions (`.github/workflows/ci.yml`)

Key changes from standard Python CI:

1. **uv instead of pip**: `astral-sh/setup-uv@v3` for dependency management
2. **Quality gate steps**: Separate named steps for each check
3. **File size enforcement**: Custom script to check 400-line limit
4. **Coverage threshold**: `--cov-fail-under=80` enforces minimum
5. **No continue-on-error**: Type checking failures block the build

### Local Pre-Commit Equivalent

Run before every commit:

```bash
#!/bin/bash
# .githooks/pre-commit or run manually

set -e

echo "🔍 Running quality gate..."

# 1. Format
echo "1️⃣ Checking format..."
uv run ruff format --check .

# 2. Lint
echo "2️⃣ Running linter..."
uv run ruff check . --fix

# 3. Security
echo "3️⃣ Running security scan..."
uv run bandit -r config/ whatsapp/ llm/ storage/ agent/ api/ utils/ -ll

# 4. Type check
echo "4️⃣ Running type checker..."
uv run mypy config/ whatsapp/ llm/ storage/ agent/ api/ utils/

# 5. File size check
echo "5️⃣ Checking file sizes..."
find . -name "*.py" -not -path "./.venv/*" -exec wc -l {} + | \
  awk '$1 > 400 {print "❌ " $2 " has " $1 " lines (max 400)"; exit 1}'

# 6. Tests
echo "6️⃣ Running tests..."
uv run pytest --cov=. --cov-fail-under=80

echo "✅ Quality gate passed!"
```

---

## Enforced Standards

| Check | Tool | Threshold | Action on Failure |
|-------|------|-----------|-------------------|
| Format | ruff | Zero violations | Block merge |
| Lint | ruff | Zero violations | Block merge |
| Security | bandit | No HIGH/MEDIUM | Block merge |
| Type Check | mypy | Zero errors | Block merge |
| File Size | wc/awk | Max 400 lines | Block merge |
| Coverage | pytest-cov | ≥80% | Block merge |

---

## Drift Detection

CI will detect deviations from Hybrid Agent System:

```
DETECTION PATTERNS:
- "pip install" in logs → WARNING: Should use uv
- "Optional[" in code → ERROR: Use X | None
- "os.path" in code → ERROR: Use pathlib
- "print(" in code → WARNING: Use logging
- Files > 400 lines → ERROR: Refactor required
```

---

## Troubleshooting

### Format Check Fails

```bash
# Auto-fix
uv run ruff format .
```

### Lint Check Fails

```bash
# Auto-fix most issues
uv run ruff check . --fix

# View remaining
uv run ruff check .
```

### Type Check Fails

```bash
# See errors
uv run mypy config/ whatsapp/ llm/ storage/ agent/ api/ utils/

# Common fixes:
# - Add type annotations to all function params/returns
# - Replace Any with specific types or object
# - Replace Optional[X] with X | None
```

### Security Check Fails

```bash
# See issues
uv run bandit -r config/ whatsapp/ llm/ storage/ agent/ api/ utils/ -ll

# Common fixes:
# - Remove hardcoded secrets (use env vars)
# - Use shell=False in subprocess
# - Use parameterized SQL queries
```

### File Size Check Fails

```bash
# Find large files
find . -name "*.py" -not -path "./.venv/*" -exec wc -l {} + | sort -n | tail -10

# Refactor approach:
# - Extract utilities to separate files
# - Split classes into multiple files
# - Move tests to co-located test files
```

### Coverage Fails

```bash
# See coverage report
uv run pytest --cov=. --cov-report=term-missing

# Add tests for uncovered lines
```

---

## Related Documents

- `HYBRID_AGENT_SYSTEM.md` — Full agent standards
- `PLANNING.md` — Project architecture (Agent Workflow Architecture section)
- `TASK.md` — TASK-025 (this workflow implementation)
- `.github/workflows/ci.yml` — Actual CI configuration

---

*Document maintained as part of TASK-025: CI/CD Alignment with Hybrid Standards*
