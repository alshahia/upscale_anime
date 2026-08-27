# V8+ MambaIRv2 + Tier 1 TODO Tracker

**Plan:** `docs/plans/v8_mambair_plan.md`
**Started:** 2026-06-20
**Last updated:** 2026-06-20
**Branch:** `perf/v7-tier1-speedups` (Tier 1 work landed here; MambaIRv2/WSL2 work deferred to user)
**Tool:** `uv` instead of `pip` (per project standard)
**Last commit (this turn):** v8 tier 1: TTA D4 8x ensemble + checkpoint souping (tests + docs)

---

## Status legend
- `[ ]` pending
- `[~]` in progress
- `[x]` done
- `[!]` blocked
- `[-]` cancelled / deferred

---

## Phase A — WSL2 uv setup

- [ ] A.1  Install `uv` in WSL2 Ubuntu-24.04 via curl installer
- [ ] A.2  `uv python install 3.11` (pin for mamba-ssm wheel compat)
- [ ] A.3  Create `.venv` in `~/projects/upscale_anime_wsl/project/.venv`
- [ ] A.4  `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- [ ] A.5  `uv pip install mamba-ssm causal-conv1d` (try pre-built wheel; fallback to source build with `uv pip install --no-binary`)
- [ ] A.6  `uv pip install -r requirements.txt && uv pip install lpips pyiqa av~=17.0 pillow-heif~=1.4`
- [ ] A.7  Sanity check: torch CUDA + mamba_ssm forward OK

## Phase B — MambaIRv2 smoke

- [ ] B.1  Download MambaIRv2 DF2K cold-start via `huggingface_hub.hf_hub_download`
- [ ] B.2  Create `configs/finetune_neosr_span_v7_mambair_smoke.yaml` with `sab_type: mamba_v2`
- [ ] B.3  Run 2-epoch smoke (`python scripts/train.py --config configs/finetune_neosr_span_v7_mambair_smoke.yaml --epochs 2`)
- [ ] B.4  Verify CUDA hot-path: `HAS_MAMBA_SSM: True`

## Phase C — Tier 1: TTA + Souping

- [x] C.1  `src/inference/tta.py` (~110 LOC; 8-element D4 group + `tta_forward`)
- [x] C.2  `scripts/inference.py --tta` flag (single-image + directory-batch paths)
- [x] C.3  `scripts/compare_checkpoints.py --tta` flag + `run_inference_tta()` helper
- [x] C.4  `tests/test_tta.py` (9 tests; D4 closure, averaging, clipping, dtype, manual-mean, CUDA smoke)
- [x] C.5  `scripts/soup_checkpoints.py` (~165 LOC; `soup_checkpoints` + `load_state_dict`)
- [x] C.6  `tests/test_soup_checkpoints.py` (8 tests; 2/3-ckpt, dtype, mismatch, fallback)

## Phase D — Verification + docs

- [x] D.1  `pytest tests/test_tta.py tests/test_soup_checkpoints.py -v` — 17 passed
- [x] D.2  `pytest tests/test_neosr_finetuner.py tests/test_fdl_loss.py tests/test_relativistic_gan.py tests/test_safe_checkpoint_load.py tests/test_two_phase_training.py tests/test_phase{2,3,4}_... tests/test_grad_scaler.py` — 128 passed across the regression slice (only pre-existing unrelated failures)
- [x] D.3  `py_compile src/inference/tta.py scripts/inference.py scripts/compare_checkpoints.py scripts/soup_checkpoints.py` — clean
- [x] D.4  `--dry-run` on `configs/finetune_neosr_span_v7_anime_smoke.yaml` — passes (1.6 GB VRAM est, 8.6 GB available)
- [x] D.5  AGENTS.md: added "V8 Tier 1 Shipped (2026-06-20)" section with file inventory + quick-start
- [x] D.6  `docs/v8_results.md` (new): Tier 1 baseline + verification + projected Δ MANIQA / LPIPS
- [x] D.7  Commit: `git commit -m "v8 tier 1: TTA D4 8x ensemble + checkpoint souping (tests + docs)"` — see commit message below.

## Phase E — User follow-up (out of scope for this workstream)

- [ ] E.1  User runs full 80-epoch MambaIRv2 finetune (after smoke passes)
- [ ] E.2  User runs souping on v6+v7 bests (after both exist)
- [ ] E.3  User runs TTA on comparison output
- [ ] E.4  User curates `data/anime_hr_holdout/` for publishable NR-IQA

---

## Sub-agent dispatch plan

### Wave 1 (parallel after Phase A+B completes)
- **Sub-agent A** → TTA implementation (C.1-C.4)
- **Sub-agent B** → Soup script (C.5-C.6)

Both agents operate on Windows. They can run in parallel with WSL2 smoke (B.1-B.4).

### Wave 2 (after Wave 1)
- Verification + docs (D.1-D.7) — single agent or main agent.

---

## Risks (from plan section E)

| Risk | Mitigation |
|---|---|
| `mamba-ssm` py3.12 wheel missing | A.2 pins Python 3.11 |
| CUDA 13.2 driver vs cu124 wheel | A.5 falls back to source build with `uv pip install --no-binary` |
| `nvcc` not in WSL2 PATH | `sudo apt install cuda-nvcc-12-4` |
| MambaIRv2 DF2K URL guess wrong | B.1 has fallback list + hf_hub search |
| MambaIRv2 forward NaN at smoke | Run with `autocast(enabled=False)` for scan; reduce LR 5x |
| TTA inv_fn wrong for `rot90` | Tests catch this |
| Soup drops `scaler_state_dict` | Documented; user must re-init scaler |