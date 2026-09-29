# Project Restructuring Plan: Clean, Structured, Scalable

> Goal: Transform the project from a collection of scripts into a proper, maintainable Python package
> that can grow beyond a single-use codebase.

---

## Current State Analysis

### Critical Structural Issues

1. **No package structure** — No `pyproject.toml`, no `setup.py`, `anime_upscaler/` has no `__init__.py`
2. **Two isolated codebases** — `src/` and `anime_upscaler/` don't share any code; GUI imports from neither
3. **Vendored code in GUI** — `archs.py` (876 lines) duplicates model code; `tta.py` is a separate implementation
4. **Root directory pollution** — 30+ markdown files, 7 backup zips, test scripts, images at root
5. **No unified entry points** — `main.py` just prints torch version; 5+ separate scripts with no CLI
6. **Configuration sprawl** — `pipeline_config.json`, `configs/`, no schema or validation
7. **Conflicting dependencies** — `requirements.txt` vs `requirements_api.txt` have different versions

### Codebase Overlap

| Component | `src/` | `anime_upscaler/` | GUI |
|-----------|---------|-------------------|-----|
| Models | ✅ span, mamba_pan, teachers | ✅ student.py, teacher.py | ✅ archs.py (vendored) |
| Inference | ✅ engine.py, tta.py | ✅ infer.py, tta.py | ✅ tta.py (vendored) |
| Training | ✅ base_trainer, model_a/b | ✅ distill.py (1076 lines) | ❌ |
| Data | ✅ augmentation, degradation | ✅ dataset.py, degradation.py | ❌ |
| Export | ❌ | ✅ export.py | ❌ |
| API | ✅ inference_api.py | ❌ | ❌ |

**Result:** 3 copies of model code, 2 copies of TTA, 2 copies of degradation, no shared utilities.

---

## Target Structure

```
upscale_anime/
├── pyproject.toml              # Package metadata, deps, entry points
├── README.md                   # Single comprehensive readme
├── LICENSE
├── .gitignore
├── .env.example                # Template for environment variables
│
├── configs/                    # All configuration (YAML)
│   ├── default.yaml            # Default config with inheritance
│   ├── training/
│   │   ├── model_a.yaml
│   │   ├── model_b.yaml
│   │   ├── ensemble.yaml
│   │   └── distill.yaml
│   ├── inference/
│   │   └── default.yaml
│   └── api/
│       └── default.yaml
│
├── src/
│   └── anime_sr/               # Single package (rename from src/)
│       ├── __init__.py
│       ├── __main__.py         # CLI entry point
│       │
│       ├── models/             # ALL model architectures
│       │   ├── __init__.py     # Registry + factory
│       │   ├── base.py         # Base model class
│       │   ├── span/           # SPAN architecture
│       │   ├── mamba_pan/      # Mamba-PAN architecture
│       │   ├── teachers/       # EDSR, RCAN, SwinIR
│       │   └── students/       # RFDN, TinySRVGGStudent
│       │
│       ├── data/               # Data pipeline
│       │   ├── __init__.py
│       │   ├── augmentation.py
│       │   ├── degradation.py  # Unified degradation
│       │   ├── datasets/       # Dataset classes
│       │   │   ├── base.py
│       │   │   ├── image.py
│       │   │   ├── video.py
│       │   │   └── multi.py
│       │   ├── preprocessing.py
│       │   └── quality.py
│       │
│       ├── training/           # Training logic
│       │   ├── __init__.py
│       │   ├── base_trainer.py
│       │   ├── trainers/       # Model-specific trainers
│       │   │   ├── model_a.py
│       │   │   ├── model_b.py
│       │   │   └── ensemble.py
│       │   ├── distillation/   # KD methods
│       │   │   ├── mtkd/
│       │   │   └── fakd/
│       │   ├── callbacks.py
│       │   ├── ema.py
│       │   └── orchestrator.py
│       │
│       ├── inference/          # Inference engine
│       │   ├── __init__.py
│       │   ├── engine.py       # Unified inference engine
│       │   ├── tta.py          # Single TTA implementation
│       │   └── tiled.py        # Tiled inference
│       │
│       ├── losses/             # Loss functions
│       │   ├── __init__.py
│       │   ├── perceptual.py
│       │   ├── pixel.py
│       │   ├── frequency.py
│       │   └── ...
│       │
│       ├── utils/              # Shared utilities
│       │   ├── __init__.py
│       │   ├── config.py       # Config loading + validation
│       │   ├── checkpoint.py   # Unified checkpoint I/O
│       │   ├── logging.py
│       │   ├── metrics.py
│       │   └── ...
│       │
│       ├── api/                # REST API
│       │   ├── __init__.py
│       │   ├── server.py
│       │   └── routes/
│       │
│       └── export/             # Export utilities
│           ├── __init__.py
│           ├── onnx.py
│           └── torchscript.py
│
├── apps/
│   ├── cli/                    # CLI application
│   │   ├── __init__.py
│   │   └── main.py             # argparse/click CLI
│   │
│   └── gui/                    # GUI application
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── controllers/
│       ├── services/
│       ├── widgets/
│       └── pipeline/
│
├── scripts/                    # Utility scripts
│   ├── export.py
│   ├── validate.py
│   ├── benchmark.py
│   └── download_models.py
│
├── tests/                      # All tests
│   ├── conftest.py
│   ├── unit/                   # Fast, isolated tests
│   │   ├── models/
│   │   ├── data/
│   │   ├── losses/
│   │   └── utils/
│   ├── integration/            # Component interaction tests
│   │   ├── training/
│   │   ├── inference/
│   │   └── api/
│   └── e2e/                    # End-to-end tests
│       ├── training_pipeline.py
│       └── inference_pipeline.py
│
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── api.md
│   ├── cli.md
│   ├── configuration.md
│   ├── development.md
│   └── deployment.md
│
└── assets/                     # Static assets
    ├── icons/
    └── examples/
```

---

## Work Packages

### WP-1: Foundation (Package Structure)

**Objective:** Create proper Python package with `pyproject.toml`, unified dependencies, and entry points.

**Tasks:**
1. Create `pyproject.toml` with:
   - Package metadata (name, version, description, authors)
   - Dependencies (unified from requirements.txt + requirements_api.txt)
   - Optional dependency groups: `[project.optional-dependencies]` dev, api, gui, mamba
   - Entry points: `[project.scripts]` anime-sr = "anime_sr.__main__:main`
   - Build system: setuptools or hatch
2. Rename `src/` → `src/anime_sr/` (proper package)
3. Add `__init__.py` to `anime_upscaler/` files (or merge into anime_sr)
4. Create `src/anime_sr/__main__.py` with CLI skeleton
5. Unify `requirements.txt` into `pyproject.toml`
6. Create `.env.example` with all env vars used in the project

**Files created:** `pyproject.toml`, `.env.example`, `src/anime_sr/__init__.py`, `src/anime_sr/__main__.py`
**Files modified:** All `src/` files (import path updates)
**Acceptance:** `pip install -e .` works; `python -m anime_sr --help` shows CLI; `import anime_sr` works

---

### WP-2: Unify Codebases

**Objective:** Merge `anime_upscaler/` into `src/anime_sr/`, eliminate vendored code.

**Tasks:**
1. Move `anime_upscaler/student.py` → `src/anime_sr/models/students/`
2. Move `anime_upscaler/teacher.py` → `src/anime_sr/models/teachers/`
3. Move `anime_upscaler/distill.py` → `src/anime_sr/training/distillation/distill.py`
4. Move `anime_upscaler/infer.py` → `src/anime_sr/inference/cli.py`
5. Move `anime_upscaler/export.py` → `src/anime_sr/export/onnx.py`
6. Move `anime_upscaler/dataset.py` → `src/anime_sr/data/datasets/image.py`
7. Move `anime_upscaler/degradation.py` → `src/anime_sr/data/degradation.py`
8. Move `anime_upscaler/tta.py` → merge into `src/anime_sr/inference/tta.py`
9. Move `anime_upscaler/validate.py` → `src/anime_sr/utils/validation.py`
10. Move `anime_upscaler/spandrel_teacher.py` → `src/anime_sr/models/teachers/spandrel.py`
11. Move `anime_upscaler/student_mambair.py` → `src/anime_sr/models/students/mambair.py`
12. Move `anime_upscaler/bench_deploy.py` → `scripts/benchmark.py`
13. Move `anime_upscaler/eval_step0.py` → `scripts/eval.py`
14. Move `anime_upscaler/eval_teachers.py` → `scripts/eval_teachers.py`
15. Move `anime_upscaler/_quick_eval_ckpt.py` → `scripts/quick_eval.py`
16. Remove vendored `apps/.../archs.py` — import from `anime_sr.models` instead
17. Remove vendored `apps/.../tta.py` — import from `anime_sr.inference.tta` instead
18. Update all imports across the project

**Acceptance:** Single source of truth for all models, inference, TTA; no vendored code; all imports resolve

---

### WP-3: Clean Root Directory

**Objective:** Move all stray files to appropriate locations.

**Tasks:**
1. Move all `*.md` files (except README.md) → `docs/`
2. Move backup `*.zip` files → `backups/` (or delete if git has history)
3. Move root `test_*.py` files → `tests/` or `scripts/`
4. Move root `*.png`, `*.jpg` files → `assets/examples/`
5. Move `pipeline_config.json` → `configs/`
6. Move `performance_benchmark_results.json` → `docs/` or `results/`
7. Move `quality_report.csv` → `results/`
8. Create `docs/architecture.md` documenting the new structure
9. Rewrite `README.md` with:
   - Project overview
   - Installation
   - Quick start (CLI, API, GUI)
   - Configuration
   - Development setup
   - Links to detailed docs
10. Create `LICENSE` file

**Acceptance:** Root directory contains only: pyproject.toml, README.md, LICENSE, .gitignore, .env.example, and directories

---

### WP-4: Unified CLI

**Objective:** Create a single CLI entry point for all operations.

**Tasks:**
1. Create `src/anime_sr/__main__.py` with subcommands:
   - `anime-sr train --config <yaml>` — Training
   - `anime-sr infer --input <path> --output <path> --model <name>` — Inference
   - `anime-sr export --model <name> --format onnx` — Export
   - `anime-sr serve --config <yaml>` — Start API server
   - `anime-sr gui` — Launch GUI
   - `anime-sr validate --model <path>` — Validate checkpoint
   - `anime-sr benchmark --model <path>` — Benchmark
2. Use `argparse` or `click` for CLI
3. Each subcommand calls the appropriate module
4. Add `--verbose`, `--quiet`, `--config` global options
5. Add shell completion support

**Acceptance:** `anime-sr --help` shows all subcommands; each subcommand works end-to-end

---

### WP-5: Configuration Management

**Objective:** Centralize and validate all configuration.

**Tasks:**
1. Create `configs/default.yaml` with all default settings
2. Create `configs/training/*.yaml` for each training mode
3. Create `configs/inference/default.yaml`
4. Create `configs/api/default.yaml`
5. Implement config inheritance (base config + overrides)
6. Add config validation (schema-based)
7. Add environment variable substitution in configs
8. Document all config options in `docs/configuration.md`

**Acceptance:** All scripts use unified config system; invalid configs raise clear errors

---

### WP-6: Test Organization

**Objective:** Categorize and clean up tests.

**Tasks:**
1. Move `tests/test_*.py` → `tests/unit/` (fast, isolated tests)
2. Move integration tests → `tests/integration/`
3. Move e2e tests → `tests/e2e/`
4. Move `apps/anime_upscaler_gui/tests/` → `tests/gui/`
5. Create `tests/conftest.py` with shared fixtures
6. Add `pytest.ini` or `pyproject.toml` [tool.pytest] section with:
   - Test paths
   - Markers (unit, integration, e2e, slow)
   - Coverage config
7. Remove duplicate tests
8. Add missing tests for new unified modules

**Acceptance:** `pytest tests/unit` runs in <30s; `pytest tests/integration` runs in <2min; `pytest tests/e2e` runs in <5min

---

### WP-7: Documentation

**Objective:** Create comprehensive, maintainable documentation.

**Tasks:**
1. `docs/architecture.md` — System design, module diagram, data flow
2. `docs/api.md` — REST API reference
3. `docs/cli.md` — CLI usage guide
4. `docs/configuration.md` — All config options
5. `docs/development.md` — Setup, testing, contributing
6. `docs/deployment.md` — Docker, cloud deployment
7. `docs/models.md` — Model architecture descriptions
8. Update `README.md` with links to all docs

**Acceptance:** New developer can set up and run the project using only docs

---

## Execution Order

```
WP-1 (Foundation) ──→ WP-2 (Unify) ──→ WP-3 (Clean Root)
                                           ↓
              WP-4 (CLI) ←── WP-5 (Config) ←──┘
                    ↓
              WP-6 (Tests) ──→ WP-7 (Docs)
```

**Phase 1:** WP-1 (Foundation) — must be first, everything depends on it
**Phase 2:** WP-2 (Unify) + WP-3 (Clean Root) — can be parallel
**Phase 3:** WP-4 (CLI) + WP-5 (Config) — can be parallel after WP-2
**Phase 4:** WP-6 (Tests) + WP-7 (Docs) — can be parallel

---

## Migration Strategy

### Risk Mitigation
1. **Git branch:** Create `refactor/restructure` branch
2. **Incremental:** Each WP is a separate commit
3. **Backward compat:** Keep old import paths working during transition (deprecation warnings)
4. **Testing:** Run full test suite after each WP

### Rollback Plan
- Each WP is a separate commit — easy to revert
- Keep `main` branch stable until all WPs are complete
- Tag current state as `v0.1-legacy` before starting

---

## Estimated Effort

| WP | Effort | Dependencies |
|----|--------|--------------|
| WP-1 Foundation | ~4h | None |
| WP-2 Unify | ~8h | WP-1 |
| WP-3 Clean Root | ~2h | WP-1 |
| WP-4 CLI | ~4h | WP-2 |
| WP-5 Config | ~3h | WP-2 |
| WP-6 Tests | ~4h | WP-2, WP-4 |
| WP-7 Docs | ~3h | All |

**Total: ~28 hours of sub-agent work, ~8 hours wall-clock with parallelism**

---

## Success Criteria

- [ ] `pip install -e .` works from clean clone
- [ ] `anime-sr --help` shows all subcommands
- [ ] `anime-sr train --config configs/training/model_a.yaml` works
- [ ] `anime-sr infer --input lr.png --output sr.png` works
- [ ] `anime-sr serve` starts API with auth
- [ ] `anime-sr gui` launches GUI
- [ ] `pytest tests/unit` passes in <30s
- [ ] Root directory has no stray files
- [ ] No vendored code (single source of truth)
- [ ] All docs are in `docs/`
- [ ] New developer can onboard using only README + docs/
