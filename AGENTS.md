# AGENTS.md — upscale_anime

Operating instructions for coding agents working in this repository. Start here; use [llms.txt](llms.txt) for the machine-readable index and [docs/AGENTS.md](docs/AGENTS.md) for the long-form internal rules (that file is deliberately excluded from the docs site).

## What this repository is

- PyTorch **anime super-resolution** toolkit plus a standalone **Windows anime upscaler GUI**.
- Two artifacts ship from one repo:
  - `apps/anime_upscaler_gui/` → `anime-upscaler-gui 0.2.0`, released as the GitHub **pre-release** `v0.2.0-preview` (wheel + `SHA256SUMS.txt`; not on PyPI).
  - root `src/anime_sr/` → `anime-sr 0.1.0`, alpha research/training package (source only; not on PyPI).
- Everything here is **preview/alpha**. Never describe it as production-ready or production-supported.

## Layout

```text
src/anime_sr/              research package: models, training, distillation, inference, API
apps/anime_upscaler_gui/   standalone Tkinter GUI (own pyproject.toml + requirements.txt)
configs/                   YAML configs (base.yaml is the root config)
scripts/                   CLI utilities, evaluation, release rehearsal
tests/                     research tests; tests/release holds artifact/packaging guards
docs/                      MkDocs Material sources (site built with mkdocs build --strict)
assets/                    example images
.github/workflows/ci.yml   CI: test, gui-tests, api-smoke, integration, performance, security (advisory), docs, release-artifacts, publish-pypi (manual-only)
```

## Setup

- Python **3.12+** for the research package; the GUI wheel supports Python 3.10+.
- Use a local virtual environment if one exists (repo `.venv` or `apps/anime_upscaler_gui/.venv`); otherwise create one.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pip install -e ".[api]"     # optional FastAPI server deps
.venv\Scripts\python.exe -m pip install -e ".[mamba]"   # optional Mamba-PAN support (may not build on Windows)
```

The GUI is developed in-tree through `apps/anime_upscaler_gui/.venv` (see [the GUI README](apps/anime_upscaler_gui/README.md)).

## Build, test, and docs commands

```bash
# Tests
pytest                                    # research tests (pyproject testpaths)
pytest apps/anime_upscaler_gui/tests -q   # GUI tests (also run in CI)
pytest tests/release -q                   # release/artifact guards — must stay green
pytest -m "not slow"                      # skip slow tests

# Docs (site: docs/ -> site/)
python -m mkdocs build --strict
python -m mkdocs serve

# Lint/format (CI: syntax lint blocking; full style/type lanes advisory)
black src/
flake8 src/
mypy src/

# Config sanity before long runs
anime-sr train --config configs/finetune_neosr_span_v7_anime.yaml --dry-run
```

## Release artifacts

- Build the GUI wheel: `pip wheel apps/anime_upscaler_gui --no-deps -w dist` (the release build uses a clean `git archive` checkout so no `__pycache__/` leaks into the wheel).
- Rehearse an installed wheel (clean-room venv, import-location assertion, sha256 + commit binding):
  `pwsh -File scripts/release/rehearse_windows.ps1 -Wheel dist/<wheel>.whl -Commit <sha>` → prints `REHEARSAL OK ... sha256=... | commit=...`.
- Root distribution (CI only): `python -m build` then `twine check dist/*`. Run it from a clean checkout: a local `build/` directory shadows the `build` module for `python -m build`.
- **Never** create tags/releases, force-push, or publish to PyPI without explicit human authorization. CI's `publish-pypi` job is manual-only (`workflow_dispatch` + `publish_pypi=true`); a `v*` tag only builds the root distribution as a CI artifact and never uploads.

## Constraints

- **Preview status.** The GUI is a pre-release; the research package is alpha. Only Windows 11 was tested for the packaged GUI (CI runs Ubuntu).
- **No PyPI package.** Do not add tag-triggered publishing, and do not claim the wheel/research package is on PyPI.
- **No bundled model weights.** Checkpoints are downloaded/imported separately; demo checkpoints are not released models.
- **Safe checkpoint loading.** `weights_only=True` is the default. Pickle fallback requires an explicit opt-in: research package `security.allow_pickle_checkpoint` / env `DSH_ALLOW_PICKLE_CHECKPOINT=true`; GUI env `ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE=1`. Only enable it for trusted files.
- **Security scan is advisory.** CI bandit reports 40 pre-existing MEDIUM findings (39 legacy `torch.load` sites + 1 `urlopen`) and is `continue-on-error`. Do not disable the job or hide findings.
- **Docs claims must match the shipped 0.2.0-preview state.** Reported benchmark numbers are project-reported single-machine measurements; link the source document and never invent numbers.
- **Docs/artifact guards.** Keep `apps/anime_upscaler_gui/tests/test_docs.py` (GUI docs links/shortcuts) and `tests/release/` green; run `mkdocs build --strict` after touching `docs/` or `mkdocs.yml`.
- **Git hygiene.** Stage exact files (never `git add -A`, `git add .`, or `git commit -a`). Commit message prefixes: `docs:`, `feat:`, `fix:`, `test:`, `ci:`. Do not rewrite history or force-push `release/gui-preview`.

## Docs and SEO files

- User docs: [README.md](README.md) (English), [README.ar.md](README.ar.md) (Arabic, RTL), [apps/anime_upscaler_gui/README.md](apps/anime_upscaler_gui/README.md).
- Citation/SEO: [CITATION.cff](CITATION.cff), [llms.txt](llms.txt), [docs/llms.txt](docs/llms.txt), [docs/citation.md](docs/citation.md).
- Agent instructions: this file, [docs/AGENTS.md](docs/AGENTS.md), [docs/index.md](docs/index.md).
- Any new page under `docs/` that should be public must be added to `mkdocs.yml` nav; otherwise `mkdocs build --strict` may still pass but the page will be unlisted.

## Where to look

- [docs/index.md](docs/index.md), [docs/training.md](docs/training.md), [docs/models.md](docs/models.md), [docs/cli.md](docs/cli.md), [docs/config_reference.md](docs/config_reference.md).
- Results and raw evidence: [docs/v7_results.md](docs/v7_results.md), [docs/v8_results.md](docs/v8_results.md), [docs/rfdn_realtime_report.md](docs/rfdn_realtime_report.md), [docs/status/MEASURED_STATUS_2026-11-16.md](docs/status/MEASURED_STATUS_2026-11-16.md).
- Upstream attribution: [apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md](apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md).
- `review_room/` is gitignored internal review material — not shipped documentation.
