# PROJECT_MEMORY.md — Live state of upscale_anime

> **The canonical "where are we?" for any session. Read this first.**
> **Update it whenever a decision is made, a phase completes, or state changes.**
> Pointed to by `AGENTS.md` "Memory protocol" section.

---

## 0. Memory protocol (for any agent)

### When to READ
- Start of every session (before any other AGENTS.md section).
- Before making non-trivial changes — confirm assumptions still hold.
- Before declaring a task complete — verify cross-references are intact.

### When to UPDATE
- A decision is made or reversed → add to "Decision log".
- A phase begins or completes → update "Phase history".
- Tasks move forward or get added/dropped → update "Pending tasks".
- A new finding (e.g. lap_var number) → update "Key empirical anchors".
- A new critical rule learned → update "Halt conditions / guardrails".

### How to UPDATE
- Append to the relevant section (do NOT rewrite history).
- Add a one-line entry with date stamp (YYYY-MM-DD).
- Keep entries short; cross-reference longer docs when needed.
- Never delete prior decisions — supersede them with a new entry that explains why.
- After updating, commit with message: `memory: <one-line summary of change>`.

### When NOT to use this file
- Long-form code documentation → `docs/plans/` or `AGENTS.md`.
- Bench tables / measurements → `docs/v7_results.md`, result JSONs.
- Per-phase detail → dedicated result doc (e.g. `student_v3_result_2026_08.md`).

---

## 1. Project snapshot

Anime super-resolution project. Two production paths:
- **RFDN Distill v1** (315K params) — real-time (~3 fps on RTX 4000), shipped, PSNR 29.89 / lap_var 21.0.
- **RealESRGAN AnimeVideo v3** (621K params, pre-trained from xinntao) — quality option, PSNR 29.04 / lap_var 58.1, already in GUI registry.

Goal: train a student that beats v1 and approaches animevideov3 quality. Phase 3 attempts (adversarial v3 RFDN) failed D.2 quality gate (lap_var 22.3 vs threshold 35). Phase 4.I1 (nearest-residual RFDN) executed 2026-09-03 — D.2 PASS but D.1 PSNR FAIL (oversharpened, 27.97 dB). Phase 4.I2 (isolated adversarial) executed same day — both sub-runs D.2 FAIL (I2a 20.8, I2b 24.2). Phase 4.I3 (SRVGG-body student) executed same day — D.2 PASS but D.1 PSNR FAIL (oversharpened, 27.91 dB, student < bicubic = H6 halt). All three NOT PROMOTED. Per plan §5 "FAIL FAIL FAIL → ship animevideov3 baseline only". **Phase 4 closed.** Phase 5 (research landscape survey) completed 2026-09-03; see docs/research/anime_sr_2026/. Phase 5#1 (warm-start v1 -> SRVGG LPIPS), 5#2 (APISR balanced twin perceptual), and 5#2b (asymmetric twin 1.0/0.25) all executed same week — none break the SRVGG-body ceiling (27.87-27.98 dB, range 0.11 dB, loss-agnostic). **Phase 5#6 (pure-PyTorch MambaIRv2 113K params) executed 2026-09-06 — D.1 FAIL (26.17 dB), D.2 PASS-trivial (lap_var 6625 hallucinated HF), H6 triggered, D.3 FAIL (32 sec/frame). NOT PROMOTED.** Architectural hypothesis (state-space breaks SRVGG-body ceiling) DISCONFIRMED at 113K capacity. Awaiting user direction on next pivot (likely 5#3 FRAMER per §6 fallback chain).

Working directory: `E:\python projects\upscale_anime`
Branch: `feature/phase-1-realtime-4k`
HEAD: see §2 below.

---

## 2. Branch and HEAD

| Commit | Message |
|---|---|
| `c82767e` | memory: Phase 4.I3 outcome in AGENTS.md + PROJECT_MEMORY.md (Phase 4 closed) |
| `7783669` | Phase 4.I3: SRVGG-body student -- D.2 PASS, D.1 FAIL, H6 triggered, NOT PROMOTED |
| `8929287` | memory: Phase 4.I2 outcome in AGENTS.md + PROJECT_MEMORY.md |
| `7ac0665` | Phase 4.I2: feat-weight flag — isolated adversarial ablation |
| `19f6fa0` | docs: reflect Phase 4.I1 outcome in AGENTS.md, plan §11, PROJECT_MEMORY.md |
| `924995f` | memory: I1 result + decision log + empirical anchors updated |
| `bd72dd3` | Phase 4.I1: nearest-residual RFDN — D.2 PASS, D.1 FAIL, NOT PROMOTED |
| `2cb458c` | memory: self-record creation in §2 HEAD table and §9 update history |
| `4434609` | memory: add docs/PROJECT_MEMORY.md (canonical session-starting state file) and AGENTS.md pointer |
| `624fb11` | AGENTS.md: add Phase 4 plan section (I1+I2+I3, awaiting approval) |
| `26ac928` | Phase 4 plan: I1 nearest-residual + I2 isolated adv + I3 SRVGG body |
| `ad14781` | Phase 3.F: AGENTS.md lessons learned + v3 result doc |

Last updated: 2026-09-03 (after Phase 4.I3 execution; **Phase 4 closed**).

---

## 3. Phase history

### Phase 1 (Real-time 4K) — SHIPPED (2026-08-31)
- TensorRT engine for RFDN student (2.37x speedup over PyTorch FP16).
- NVENC hardware encoder (26.5% CPU vs 42.5% libx264).
- Batching deferred (RFDN too small; b4 is slower than b1).
- **End-to-end fps**: 18.80 mean, 20.52 peak at 854x480 to 3416x1920.
- Goal: >=25 fps — **PARTIAL** (~75% of target).
- Full report: `docs/rfdn_realtime_report.md` Section 10.

### Phase 2 v3 student — SHIPPED (predecessor of v3)
- SPAN-teacher distillation, RFDN 315K, 40 epochs LR=5e-5.
- PSNR 29.89, lap_var 21.0.
- This is the current `RFDN_distill_v1_4x_student.pth`.

### Phase 3 v3 (adversarial) — NOT PROMOTED (2026-09-02)
- RFDN warm-start + animevideov3 teacher + PatchGAN70 + HingeGANLoss + edge loss.
- 30 epochs, ~17 min on RTX 4000.
- Result: PSNR 29.99 (marginal +0.1 dB over v1), **lap_var 22.3** (D.2 gate FAILED, threshold 35).
- Two warm-start + shortcut-anneal runs mode-collapsed (PSNR 29.93 to 5.41 in 5 ep).
- **Lesson**: shortcut anneal is hostile to warm-start. Adversarial alone plateau.
- Ckpt preserved un-promoted: `runs/distill_v3_4x_v3_epoch18_ema_unpromoted.pth`.
- Full doc: `docs/plans/student_v3_result_2026_08.md`.

### Phase 4 — I1 + I2 + I3 EXECUTED, ALL NOT PROMOTED (2026-09-03)
- Three sequential experiments (I1 → I2 → I3) to break the RFDN+anneal ceiling.
- **I1**: nearest-residual RFDN — **COMPLETE**, **NOT PROMOTED**.
  - Hypothesis "nearest residual → sharper" confirmed: full-frame lap_var **109.4** (5.2× v1).
  - But PSNR regressed: **27.97 dB** vs v1 29.89 (−1.92 dB), student < bicubic (−1.48 dB).
  - D.2 gate PASS (lap_var ≥ 35). D.1 gate FAIL (PSNR ≥ 29.0). Sharpness ≠ quality.
  - Likely root cause: adversarial + from-scratch + new-shortcut = oversharpening.
  - Run preserved: `runs/distill_i1_nearest_residual/` (per Q6 rule).
  - Full doc: `docs/plans/student_phase4_i1_nearest_residual_result.md` (9056 bytes).
- **I2**: isolated adversarial — **COMPLETE**, **BOTH sub-runs NOT PROMOTED** (D.2 FAIL).
  - Two sub-runs warm-started from v1: I2a animevideov3 + adv + feat=1 (30 ep); I2b SPAN + adv + **feat_weight=0** (40 ep).
  - **I2a (sanity check, identical to Phase 3 v3 recipe)**: val PSNR EMA 29.885 (= v1); full-frame lap_var **20.8** (= v1). Animevideov3+adv+feat=1 is a no-op over v1.
  - **I2b (the new isolation)**: val PSNR EMA **29.911** (+0.022 dB over v1); full-frame lap_var **24.2** (+15% over v1). SPAN+adv+feat=0 yields a small but real improvement but still below 35 threshold.
  - **Lesson**: SPAN is a stronger distillation teacher than animevideov3 for RFDN. `feat_weight=0` is the safe default for any future SPAN+adv RFDN variant. Animevideov3's per-pixel response is too lossy for adversarial-only intervention to help; SRVGG body (I3) is needed.
  - Runs preserved: `runs/distill_i2a_adv_only_animevideov3/`, `runs/distill_i2b_adv_only_span/` (per Q6 rule).
  - Full doc: `docs/plans/student_phase4_i2_isolated_adv_result.md` (10196 bytes).
- **I3**: pending user decision. Per plan §5 "FAIL FAIL → ship animevideov3 baseline only"; I3 is optional.
- Full plan: `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` (391 lines).
- Code added (Phase 4 I1): `--shortcut-mode {bicubic,nearest}` arg, `RFDN(shortcut_mode=...)` constructor, `set_shortcut_mode()` method, conditional `align_corners` for nearest mode, `tests/test_shortcut_mode.py` (4 tests, all passing).
- Code added (Phase 4 I2): `--feat-weight {float, default=1.0}` arg, gate at line 539 (feature distillation block conditioned on `args.feat_weight > 0`), weight multiplier at line 552. All 19 pre-existing tests still pass. Default = 1.0 (backward-compatible with Phase 2 v3 / Phase 3 recipe).
- Code added (Phase 4 I3): new `TinySRVGGStudent` class in `student.py` (52 ch × 12 convs + PReLU + PixelShuffle + nearest residual, 317,300 params; mirrors animevideov3's SRVGGNetCompact exactly), `--arch {rfdn,srvgg}` argparse in `distill.py` with arch-aware build + forward dispatch (s_feats=[] when srvgg), vendored `TinySRVGGStudent` + vendored RFDN shortcut_mode honor in `apps/.../archs.py::build` (sniffs num_feat/num_conv/scale from srvgg_student ckpt; auto-sniffs shortcut_mode from rfdn_student ckpt args), 9 new smoke tests in `tests/test_tiny_srvgg.py` (all passing). All 28 pre-existing tests still pass. Default `--arch rfdn` (backward-compatible with Phase 2 v3 / Phase 3 / Phase 4 I1+I2 recipes).

### Phase 5 (Research) — Landscape survey COMPLETED (2026-09-03)
- **Scope:** User asked to research latest anime SR techniques/models for highest quality AND real-time capable (>=25 fps @ 4K), including non-real-time options for hybrid approach. Documented as research progress.
- **Method:** 15 Exa /search queries via exa-py 2.20.0 (Python SDK). num_results=8, type=auto, contents.highlights (numSentences=4, highlightsPerUrl=3). Total cost ~$0.13. No deep-crawl.
- **Key findings:**
  - **Real-time (<=25 fps @ 4K):** Lightweight CNN family remains dominant (RFDN, IMDN, BSRN, ShuffleMixer, LKDN, VAPSR, OmniSR, ELSANet, ESDAN). Our v1 RFDN is a strong, mature baseline. TensorRT FP16 + batched video is the biggest latency lever.
  - **Quality ceiling:** Diffusion SR (SUPIR, SeeSR, DiffBIR, StableSR, OSEDiff, SinSR) at 5-15 sec/frame on RTX 4090. **One-step variants** (OSEDiff, SinSR, FiDeSR, One-Step Diffusion Transformer, Bridging Fidelity-Reality, AlloSR^2 — all CVPR 2026) close gap to ~0.5-1 sec/frame.
  - **Latest breakthrough (CVPR 2025/2026):** **MambaIRv2** matches/beats SwinIR-class transformers at 30% lower MACs, +0.29 dB on Manga109. Strong candidate but requires Linux+cu124/cu126 mamba-ssm wheel — Windows requires WSL2.
  - **Anime-specific SOTA:** animevideov3 + Real-CUGAN + Anime4K v4 remain mainstays. **APISR (CVPR 2024)** is now the strongest open-source anime recipe (API dataset + balanced twin perceptual loss).
  - **Knowledge distillation advances (2024-2026):** Multi-Scale Contrastive-Adversarial (ICCVW 2025), FRAMER frequency-aligned self-distillation (Dec 2025), Distillation-Supervised ConvLoRA (2025), Multi-Granularity Mixture of Priors (2024). None integrated into our recipe yet.
  - **GAN training:** APISR balanced twin perceptual loss + MSA-ESRGAN multi-scale U-Net discriminator (Sci.Rep 2024) + DPO-ESRGAN (MDPI 2025).
  - **Cascade 2x+2x:** Confirmed by CASR (CVPR 2024 workshop) and CARN (ECCV 2018) as the standard real-time 4K approach — validates our existing Phase 2 plan.
- **Hybrid recommendation:** Three-tier GUI: real-time (RFDN + TensorRT) + quality (OSEDiff/SinSR 1-step anime fine-tune) + hybrid (cascade 2x+2x).
- **Decision matrix (REPORT.md §9) lists 9 candidate next experiments** ranked by effort/risk/quality-gain. None committed yet — awaiting user direction.
- **Docs:** `docs/research/anime_sr_2026/` (README.md index, REPORT.md 22.7 KB / 358 lines, SOURCES.md 51 KB / 447 lines, exa_search_results.json 581 KB raw data).
- **Security note:** User pasted Exa API key directly in chat; key NOT stored in any committed file. Recommend rotation at https://dashboard.exa.ai/keys after this session.

### Phase 5#6 (MambaIRv2 pure-PyTorch) — EXECUTED, NOT PROMOTED (2026-09-06)
- **Pivot rationale**: The 4-run SRVGG-body ceiling (I3, 5#1, 5#2, 5#2b all 27.87-27.98 dB) suggested loss tuning was exhausted and an architecture change was needed. MambaIRv2 was the top CVPR 2025/2026 candidate. Docker build (Linux+cuda126 mamba_ssm wheel) was 60-90 min commitment with too many unknowns; the user pivoted to **pure-PyTorch** so we could test the architectural hypothesis (does state-space break the SRVGG-body ceiling?) without infrastructure risk.
- **Code added**: `anime_upscaler/student_mambair.py` (16 KB, pure-PyTorch MambaIRv2-style VSSBlock with 4-direction vectorised selective scan via parallel-prefix trick + cum-clamp ±20). `--arch {rfdn,srvgg,mambair}` in distill.py; student build branch at line 478-494; `--mambair-embed-dim/-num-blocks/-d-state` CLI args. Defaults: E=48, NB=8, d_state=16 → 113K params (36% of SRVGG 317K). 5 surgical edits to distill.py, all backward-compatible (`--arch` default rfdn).
- **Single-fix OOM**: First two training runs OOMed at SS2D scan (held SS2D's (B,L,d_inner,d_state) intermediates across all 8 blocks = ~5 GB peak on Quadro RTX 4000). Fix: gradient checkpointing on VSSBlock (gated on `self.training`, so eval pays no recompute cost). Reduces training-time peak from OOM (~7.8 GB used) to 3.6 GB (fits in 8 GB with 4 GB headroom).
- **Pilot run** (10 epochs, batch=16, animevideov3 teacher, no adv, lambda-adv=0, val-batches=4, 23 min wall):
  - E=48, NB=8, d_state=16, **113,952 params** (36% of SRVGG 317K)
  - Val trajectory (raw PSNR): ep1 26.47 → ep4 27.62 → ep5 **27.63** (PEAK) → ep6 27.56 → ep7 27.47 → ep10 27.37 — clear plateau + regression at ep5-6, signature identical to SRVGG family
  - Val EMA PSNR: 25.37 → 26.51 (climbing slowly; α=0.999 EMA hasn't converged at 10 ep)
  - loss_total descending (0.463 → 0.321), SSIM ascending (0.6338 → 0.7551)
  - **Per-Q6 run preserved**: `runs/distill_mambair_v1_10ep_v3/` (10 epoch ckpts + train_log.csv)
- **Eval results** (using `student_best_ema.pt` from epoch 4, the best-by-val-psnr ckpt):
  - `tmp/eval_mambair_test_psnr.py` (NEW, ~140 LOC, mirrors `scripts/eval_v3_ckpt.py` for mambair): n=99 test split, batch=8, 20 s
    - bicubic 29.53 / **student 26.17** / teacher 29.09 (SSIM 0.8796 / 0.6289 / 0.8890)
    - **D.1 FAIL (-2.83 dB gap)**; **H6 TRIGGERED** (student -3.36 dB below bicubic)
  - `tmp/eval_mambair_full_frame.py` (NEW, ~110 LOC, mirrors `tmp/eval_i3_full_frame.py` for mambair): full-frame 854×480 → 3416×1920 @ 30 frames
    - bicubic 11.79 / **student 6625.19**
    - latency **32,398 ms/frame** (pure-PyTorch scan is 405× over 80 ms D.3 budget)
    - **D.2 PASS** trivially (185× threshold), BUT meaningless: PNG entropy 5.5× bicubic = hallucinated HF noise; lap_var at this magnitude is artifact, not detail
    - **D.3 FAIL** — same conclusion as I3 timing numbers showed SRVGG at 61ms; pure-PyTorch Mamba at 32s is fundamentally incompatible with real-time
- **Five-run structural ceiling is now EXTENDED to 5 independent configurations — and CONFIRMED**: Mamba 113K from-scratch at 26.17 dB **under-performs** SRVGG family at 317K (27.87-27.98 dB). Architectural hypothesis at this capacity is **DISCONFIRMED**: state-space inductive bias did not break the ceiling, and Mamba additionally performs WORSE than SRVGG in this regime — likely due to insufficient capacity (113K vs 317K, ~3× capacity gap) combined with insufficient training (10 ep vs 40) and pure-PyTorch scan inefficiencies that may degrade gradient quality.
- **Decision**: Abandon pure-PyTorch Mamba path. The CUDA mamba_ssm path would require 60+ min Docker build and only matters if a much larger Mamba (~2.6M params, ~30 hours training per the MambaIRv2 paper) breaks the ceiling — not a commitment worth making on speculative gains. **Pivot to 5#3 (FRAMER-style frequency-domain distillation)**: additive loss change, may help edge preservation without changing the SRVGG body backbone. ~1-2 weeks effort.
- **Code/files added this phase**:
  - `anime_upscaler/student_mambair.py` (NEW, ~16 KB): MambaIRv2-style student + vectorised scan + grad checkpointing on VSSBlock
  - `tmp/eval_mambair_test_psnr.py` (NEW, ~140 LOC): test-split PSNR/SSIM harness for mambair ckpts
  - `tmp/eval_mambair_full_frame.py` (NEW, ~110 LOC): full-frame lap_var + latency harness for mambair ckpts
  - `tmp/mambair-build/` (preserved per Q6): earlier Dockerfile attempt + dry logs (NOT deleted, may be useful for a future Docker-based cuda126 mamba_ssm build if user changes mind)
  - `anime_upscaler/distill.py` (5 surgical edits: arch choice, runtime guard, build branch, eval branch, 3 CLI args) — all backward-compatible

---

## 4. Pending tasks

### Blocked on user re-approval (matrix "FAIL FAIL → ship animevideov3 baseline only")

- [x] **Phase 4.I1** — Nearest-residual RFDN. **COMPLETE, NOT PROMOTED** (D.1 FAIL).
  Docs: `docs/plans/student_phase4_i1_nearest_residual_result.md`. Per Q6, run preserved.
- [x] **Phase 4.I2** — Isolated adversarial. **COMPLETE, NOT PROMOTED** (both D.2 FAIL).
  Code: ~5 LOC change (--feat-weight flag + gate + multiplier), backward-compat default=1.0.
  Two sub-runs: I2a (animevideov3 + adv, val PSNR EMA 29.885 = v1, lap_var 20.8),
  I2b (span + adv + feat=0, val PSNR EMA 29.911 = +0.022 over v1, lap_var 24.2).
  Per Q6, both runs preserved under `runs/distill_i2a_*/` and `runs/distill_i2b_*/`.
  Docs: `docs/plans/student_phase4_i2_isolated_adv_result.md`.
- [x] **Phase 4.I3** — SRVGG-body student. **COMPLETE, NOT PROMOTED** (D.2 PASS, D.1 FAIL, H6 triggered).
  Code: ~+320 LOC total (new `TinySRVGGStudent` class in `student.py`, `--arch {rfdn,srvgg}` arg +
  dispatch in `distill.py`, vendored GUI copy of `TinySRVGGStudent` + auto-sniff `shortcut_mode` in
  `archs.py::build`, 9 new smoke tests in `tests/test_tiny_srvgg.py`). PREREQUISITE: vendored
  `archs.py:591` STALE shortcut_mode (PROJECT_MEMORY §8 line 299) — fixed as part of I3.
  One run, 40 ep, ~21 min, val PSNR EMA **28.10**, held-out test PSNR **27.91**, full-frame lap_var
  **306.48** (5.3× animevideov3, 14.6× v1). Latency 61 ms/frame (1.7× faster than v1 RFDN).
  H6 oversharpening halt triggered: student 27.91 < bicubic 29.45. Architecture unlocks
  animevideov3's response signal but overshoots by 5× even at adv=0 (epoch 1 in-batch lap_var 3236).
  Per Q6, run preserved: `runs/distill_i3_srvgg_body/` (40 epochs, last 5 un-archived).
  Docs: `docs/plans/student_phase4_i3_srvgg_body_result.md` (~13 KB).



### Phase 5 candidates -- SORTED BY ROI (best quality gain / wall-time)

Ranking criterion: ROI = (expected PSNR + lap_var + perceptual gain) / (effort + risk + infra friction). Full analysis in docs/research/anime_sr_2026/RECOMMENDED_PATH.md.

- [x] **5#1 (2026-09-03): Warm-start v1 -> SRVGG-body + no-adv + LPIPS-heavy** (S, 1-2 days) -- reuses I3 code. EXECUTED 2026-09-03: D.1 FAIL (test PSNR 27.87 dB < 29.0), D.2 SKIPPED, H6 triggered. SRVGG body sharpness bias persists despite warm-start + no-adv. NOT PROMOTED.
- [x] **5#2 (2026-09-05): APISR balanced twin perceptual loss** (S, 1 day) -- --loss twin flag wired into distill.py (ImageNet-only BSD-3 fallback when Danbooru weights absent; see src/losses/twin_perceptual_loss.py:138-139). Reuses src/losses/twin_perceptual_loss.py already in repo + tests/test_v7_loss_schedule.py::TestDanbooruVGGWeightBalance. EXECUTED 2026-09-05: D.1 FAIL (test PSNR 27.98 < 29.0, gap -1.02 dB), but +0.11 dB over Rank #1 (27.98 vs 27.87). SRVGG body sharpness bias is STRUCTURAL -- same ceiling across 3 independent runs (I3 from-scratch 27.91, Rank #1 27.87, Rank #2 27.98). NOT PROMOTED.
- [x] **5#2b (2026-09-05): Asymmetric twin 1.0/0.25** (S, ~25 min) -- fallback (e) from 5#2 result. Lean on anime-domain ResNet50 signal alone (drop photo VGG19 weight 4x). Same recipe as 5#2 but --twin-danbooru-weight 1.0 --twin-vgg-weight 0.25. Output: runs\distill_v3_4x_srvgg_twin_asym. EXECUTED 2026-09-05 (job pwsh-7): D.1 FAIL (test PSNR 27.87 < 29.0, gap -1.13 dB); ties Rank #1 and regresses -0.11 dB vs Rank #2. SRVGG body ceiling is loss-agnostic; +0.11 dB Rank #2 gain was from BALANCED twin regularization (VGG19 was contributing), not from ResNet50 alone. NOT PROMOTED. See decision log.
- [ ] **5#3: FRAMER-style frequency-domain distillation** (M, 1-2 weeks) -- frequency-aligned self-distillation; +HF detail. Low risk.
- [ ] **5#4: Multi-Scale Contrastive-Adversarial KD** (M, 1 week) -- ICCVW 2025 method; +LPIPS.
- [ ] **5#5: OSEDiff / SinSR anime fine-tune** (L, 2 weeks) -- 1-step diffusion as quality-tier preset; +++ MANIQA. Heavy compute.
- [x] **5#6 (2026-09-06): MambaIRv2 student distillation — PURE-PyTorch path** (M, 1 day setup + 23 min pilot) -- pivoted from Docker build (mamba-ssm wheel unbuildable on Windows native; WSL2+Docker was 60-90 min commitment with too many unknowns). New `anime_upscaler/student_mambair.py` (16 KB, 113K params @ E=48 NB=8 d_state=16), `--arch mambair` wired into distill.py (5 surgical edits, backward-compat default rfdn). Pilot 10-epoch run at E=48 NB=8 with grad-checkpointing on VSSBlock (5.9 GB peak @ train B=16). EXECUTED: D.1 FAIL (test PSNR **26.17 dB**, gap -2.83 dB), D.2 PASS-only-trivially (full-frame lap_var 6625 — 21× I3's 306, but PNG entropy 5.5× bicubic = hallucinated HF noise, NOT real detail), H6 triggered (student -3.36 dB below bicubic), D.3 FAIL (32,398 ms/frame @ 854×480 → 3416×1920 — pure-PyTorch scan is 405× over 80 ms budget). **NOT PROMOTED.** Architectural hypothesis DISCONFIRMED at 113K capacity: state-space inductive bias did not break SRVGG-body ceiling; Mamba tracked same structural sharpness bias, plus 32-sec inference. Five-run ceiling confirmed across 5 independent runs (I3 from-scratch, 5#1, 5#2, 5#2b, 5#6 — range now 26.17-27.98 dB, **weighted arch ceiling is Mamba under-performs SRVGG family in this regime**). Output: `runs/distill_mambair_v1_10ep_v3/` (preserved per Q6). Full doc: `docs/plans/student_phase5_mambair_result.md`. **Decision: abandon pure-PyTorch Mamba path. CUDA mamba_ssm swap is gated on Docker (60+ min) and would only be relevant if a much-larger Mamba (2-3M params, 30+ hours train) breaks the ceiling — not worth the commitment. Pivot to 5#3 FRAMER (frequency-domain distillation, additive loss, may help without changing backbone).**
- [ ] **5#7: FiDeSR / One-Step Diffusion Transformer anime fine-tune** (L, 3 weeks) -- CVPR 2026 SOTA; ++++ perceptual. Quality tier only.


### Always

- [ ] Update this file whenever state changes.

---

## 5. Key empirical anchors

### Anime video test (854x480 to 3416x1920, 18 frames, ~1 s clip)

| Model | PSNR (test dB) | lap_var (full frame) | SSIM | fps | infer ms |
|---|---|---|---|---|---|
| bicubic | 29.45 | 11.8 | 0.8780 | — | 22 |
| v1 RFDN (315K, ours) | **29.89** | 21.0 | 0.87 | 3.07 | 104.4 |
| v3 RFDN (315K, failed) | 29.99 | 22.3 | — | 3.22 | 102.4 |
| **RealESRGAN AnimeVideo v3** (621K, xinntao) | 29.04 | **58.1** | 0.8863 | 3.21 | 98.7 |
| SPAN checkpoint (2.2M, ours) | 30.44 (test retrain) | — | 0.9218 | — | — |
| **Phase 4.I1 RFDN (315K, nearest)** | 27.97 (regressed) | **109.4** (5× v1, oversharpened) | 0.8335 | — | 86.5 | D.2 PASS, D.1 FAIL, **NOT PROMOTED** |
| **Phase 4.I2a (RFDN 315K, animevideov3+adv+feat=1)** | 29.32 (test) | **20.8** (= v1) | — | — | 95.8 | D.2 FAIL (= v1, no improvement) |
| **Phase 4.I2b (RFDN 315K, span+adv+feat=0)** | 29.35 (test) | **24.2** (+15% over v1) | — | — | 97.2 | D.2 FAIL (+15%, still below 35) |
| **Phase 4.I3 SRVGG (317K, animevideov3+adv)** | 27.91 (test) | **306.5** (5.3× animevideov3, 14.6× v1) | 0.8508 | — | **61.0** | D.2 PASS, D.1 FAIL, H6 TRIGGERED, **NOT PROMOTED** |
| **Phase 5#6 Mamba (113K, animevideov3+no-adv, pure-PyTorch)** | 26.17 (test) | **6625.2** (21× I3, 296× v1 — but PNG entropy 5.5× bicubic = hallucinated HF) | 0.6289 | 0.031 (32s/frame) | **32,398** | D.2 PASS-trivial, D.1 FAIL, H6 TRIGGERED, D.3 FAIL (405× over budget), **NOT PROMOTED** |

> Important: the **27.97** for I1 comes from distill.py's `evaluate()` on the held-out `test` split of `anime_video_frames` (192×192 crops). The PSNR for v1/v3/animevideov3 in this table is the **clip** mean from the canonical harness. Same data domain, slightly different test pipeline — close enough to flag the regression but not directly comparable. The **lap_var 109.4** was measured with `tmp/eval_i1_correct_shortcut.py` on the same `tmp/real_video_1sec.mp4` frame 8 used by the canonical harness, so it IS directly comparable to v1's 21.0 / animevideov3's 58.1.

### Architecture comparison

| Model | Params | Residual type | Block design |
|---|---|---|---|
| RFDN v1 (ours) | 315K | **bicubic** | FIM + PixelAttention |
| SRVGG animevideov3 | 621K | **nearest** | Plain 3x3 convs + PReLU |
| SPAN ckpt (ours) | 2.237M | **bicubic** | Swift Parameter-free Attention |

**Key insight (2026-09-03)**: SPAN checkpoint is ALSO bicubic-residual — bicubic isn't the sole reason for v1's smoothness. Animevideov3's nearest residual + anime-specific training data explains most of the gap.

**Phase 5#6 addition (2026-09-06)**: MambaIRv2-style 4-direction visual state-space (VSS) blocks with selective scan are ANOTHER inductive bias variant. At pure-PyTorch 113K params + 10 epochs + no adversarial, Mamba tracks the SRVGG-body structural ceiling (same regression curve) and under-performs SRVGG (26.17 vs 27.91 dB test). State-space modeling alone does NOT break the ceiling at this capacity — likely needs ≥3× capacity (MambaIRv2 paper baseline is ~2.6M params) and CUDA mamba_ssm kernels (pure-PyTorch scan is 405× too slow for inference). **Conclusion**: inductive bias change (CNNs vs SSMs) is not the binding constraint at this scale; the binding constraint is the 192×192 training patch + 240-pair dataset + 0.5 L1 distillation recipe.

### Failure signatures (so we don't retry dead paths)

- From-scratch + shortcut-anneal + LR >= 2e-4 → mode collapse (PSNR drops to 5 dB in 5 ep).
- Warm-start v1 + shortcut-anneal → mode collapse (PSNR 29.92 to 5.41).
- Warm-start v1 + span + adv + feat=1 → regression (PSNR 28.91 < v1's 29.89).
- Adversarial alone (animevideov3, no feat) → marginal only (PSNR 29.99, lap_var 22.3).
- **NEW 2026-09-03**: From-scratch + nearest-residual + adv=0.001 → oversharpened
  (PSNR 27.97 < bicubic 29.45; lap_var 109.4, **5× v1 but 1.9× animevideov3**).
  Adversarial + from-scratch + new shortcut together = oversharpening.
  Pick one new thing per run; do NOT combine all three.
- **NEW 2026-09-03**: From-scratch + new-arch (SRVGG) + adv=0.001 → oversharpened
  (PSNR 27.91 < bicubic 29.45; lap_var 306.5, **14.6× v1, 5.3× animevideov3**).
  Adversarial + from-scratch + new architecture together = oversharpening.
  Same failure signature as I1 with a different "new thing" (arch vs shortcut).
  **Generalization**: from-scratch + any single design choice + adversarial = overshoot.
  Either warm-start the body, or drop adversarial, or push LPIPS/GT anchor weight up.

---

## 6. Decision log

### 2026-09-03 — Phase 4 plan committed but NOT executing
- Decision: Wait for user approval before modifying code.
- Reason: Phase 3 already attempted 4 variations and all plateaued at lap_var 22-23. The user requested a plan, not unilateral execution.
- Reversal cost: low (plan is reversible via git checkout).

### 2026-09-03 — Phase 4.I1 executed (user approved after seeing the plan)
- Decision: Run `--shortcut-mode nearest` from-scratch per plan §1.3.
- Reason: User said "go to Phase I1".
- Result: D.2 PASS (lap_var 109.4), D.1 FAIL (PSNR 27.97). NOT PROMOTED.
- Reversal cost: low. Code change is backward-compatible (default = bicubic).

### 2026-09-03 — Sharpness ≠ quality (lesson from I1)
- Decision: Treat PSNR ≥ 29.0 dB as the binding constraint for any future student,
  not lap_var ≥ 35.
- Reason: I1 lap_var 109.4 (5× v1) was a great sharpness number but PSNR dropped
  −1.92 dB below v1. A model can trivially maximize lap_var by hallucinating edges.
- Action: All future Phase 4/5 promotions require BOTH D.1 AND D.2 (D.1 binds).

### 2026-09-03 — Animevideov3 is shipped as quality option (carry-over from Phase 3.E)
- Decision: RealESRGAN AnimeVideo v3 (pre-trained, xinntao) ships in GUI as quality option.
- Reason: Cannot be matched by our RFDN student at available compute (~17 min/run).
- Registry: `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG`.

### 2026-09-02 — RFDN+anneal ceiling confirmed (carry-over from Phase 3 lessons)
- Decision: Do NOT warm-start RFDN with shortcut-anneal.
- Reason: Three separate runs mode-collapsed (PSNR drops to 5-7 dB).
- Rule: warm-starting → `--shortcut-anneal off`; from-scratch → `--shortcut-anneal 1to0` OK.

### 2026-09-02 — Adversarial+feature-distillation is a bad combo
- Decision: Avoid combining L_adv with L_feat without isolation.
- Reason: span+adv+feat=1 regressed (28.91 dB); the two objectives pull in opposite directions.
- Action: Phase 4.I2 will isolate via `--feat-weight` flag.

### 2026-09-03 — Phase 4.I2 executed (user approved after seeing plan + I1 result)
- Decision: Run both sub-runs per plan §2.3: I2a animevideov3+adv+feat=1 (30 ep, sanity),
  I2b SPAN+adv+feat=0 (40 ep, the new isolation).
- Reason: User said "Proceed to I2 (isolated adversarial)". The plan §2.1 hypothesis was
  that L_feat+L_adv was the regression driver; isolating them reveals which is responsible.
- Result: I2a D.2 FAIL (lap_var 20.8 = v1, no improvement); I2b D.2 FAIL but PARTIAL WIN:
  val PSNR EMA 29.911 = +0.022 dB over v1, full-frame lap_var 24.2 = +15% over v1. Both NOT
  PROMOTED per matrix "FAIL FAIL → ship animevideov3 baseline only".
- Reversal cost: low. Code change is backward-compatible (`--feat-weight` defaults to 1.0,
  matches Phase 2 v3 / Phase 3 recipe). All 19 pre-existing tests pass.

### 2026-09-03 — feat_weight=0 unlocks SPAN+adv gradient (lesson from I2b)
- Decision: For any future SPAN+adv RFDN variant, default `--feat-weight 0` (disable L_feat).
- Reason: I2b reproduces the 2026-08 span+adv recipe but drops L_feat; goes from
  28.91 dB regression (span+adv+feat=1) to +0.022 dB over v1 (span+adv+feat=0). The
  L_feat L2-normalized cosine pull on SPAN's intermediate features was actively hurting
  the student when combined with adversarial's distribution-matching gradient.
- Action: Use `--feat-weight 0` in any future variant. Animevideov3+adv path remains a
  no-op (I2a = v1); the student needs SRVGG body (I3) to make animevideov3's response
  signal useful at this parameter scale.

### 2026-09-03 — Animevideov3 is the GRAIN OF TRUTH for shipping (re-affirmed post-I2)
- Decision: Animevideov3 baseline remains the canonical "quality" GUI option.
- Reason: Cannot be matched by RFDN student at available compute. After 4 distinct
  attempts (Phase 3 v3 + Phase 4 I1/I2a/I2b), the best RFDN variant (I2b) still lands at
  29.91 dB val PSNR / 24.2 lap_var — far short of animevideov3's 58.1 lap_var. The
  remaining 2.4× sharpness gap is the SRVGG-body contribution (I3 still pending if
  user wants to attempt it).
- Registry: `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG`.

### 2026-09-03 — Phase 4.I3 executed (user approved after seeing I2 outcome)
- Decision: Run SRVGG-body student per plan §3.4: `--arch srvgg --teacher animevideov3
  --lambda-adv 0.001 --shortcut-anneal off --epochs 40` (single run, no warm-start).
- Reason: User said "Proceed to I3 (SRVGG body student)". Architecture change is the
  only path left after I1 + I2 both failed the binding gates. Plan §3.1 hypothesis:
  SRVGG body matches animevideov3's inductive bias exactly.
- Result: **D.2 PASS (lap_var 306.5, 5.3× animevideov3 — biggest sharpness jump ever)**
  but **D.1 FAIL (PSNR 27.91 < bicubic 29.45 → H6 oversharpening halt triggered)**.
  NOT PROMOTED. Per plan §5 matrix "FAIL FAIL FAIL → ship animevideov3 baseline only".
- Reversal cost: low. All Phase 4 code is backward-compatible: `--arch rfdn` is the
  default (preserves Phase 2 v3 / Phase 3 / Phase 4 I1+I2 behavior); vendored
  `archs.py` change is forward-compatible (default bicubic for v1 ckpts).
- Latency win: 61 ms/frame vs v1's 104 ms (1.7× faster). If a future SRVGG student
  variant achieves PSNR ≥ 29.0, this latency headroom is on the table.

### 2026-09-03 — SRVGG body unlocks animevideov3's signal but overshoots (lesson from I3)
- Decision: For any future SRVGG-student variant, do NOT combine from-scratch + adv.
  Either warm-start the body from v1 (no from-scratch), or drop adversarial
  (replace with stronger GT anchor: LPIPS weight 1.0 instead of 0.05).
- Reason: I3's architecture change DID unlock animevideov3's response signal — full-frame
  lap_var 306 vs animevideov3's 58, in-batch lap_var 3236 at epoch 1 (adv=0!). The
  SRVGG body has a sharpness prior that produces high-frequency outputs from random init.
  When combined with adversarial's distribution-matching gradient, the student overshoots
  by 5× even when adversarial weight is just 0.001. The pixel accuracy loss (−1.98 dB vs
  v1) is the cost; the sharpness gain is "free" but unwanted.
- Action: Phase 5 candidate experiments (warm-start v1 → finetune with SRVGG-distill;
  or SRVGG + no-adv + LPIPS-weight=1.0). Both orthogonal fixes from I3's recipe.

### 2026-09-03 — Animevideov3 + v1 RFDN remain shipping (re-affirmed post-I3)
- Decision: Phase 4 closed. v1 RFDN + animevideov3 SRVGG remain the only production
  GUI options.
- Reason: All three Phase 4 sub-experiments failed the binding D.1 (PSNR ≥ 29.0) gate
  per plan §5 "FAIL FAIL FAIL → ship animevideov3 baseline only". SRVGG body is the
  correct architecture (latency win confirmed) but at this compute / training budget
  (17-22 min/run) cannot be made PSNR-accurate enough to ship.
- Registry: `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG`.

### 2026-09-03 -- Phase 5 research survey committed as decision-input document
- Decision: Landscape research documented at docs/research/anime_sr_2026/ (README + REPORT + SOURCES + raw JSON). NOT an implementation plan; no code changes.
- Reason: User asked to research latest anime SR techniques/models (real-time + quality + hybrid). Documented as progress.
- Method: 15 Exa /search queries (exa-py 2.20.0) at ~0.13 USD total. No deep-crawl. All citations linked in SOURCES.md.
- Key new external models to consider: MambaIRv2 (CVPR 2025) +0.29 dB on Manga109 vs HAT, OSEDiff (NeurIPS 2024) 1-step diffusion, FiDeSR / One-Step Diffusion Transformer (CVPR 2026) one-step diffusion SOTA.
- Key new methods to consider: APISR balanced twin perceptual loss (CVPR 2024), FRAMER frequency-aligned self-distillation (Dec 2025), Multi-Scale Contrastive-Adversarial KD (ICCVW 2025).
- Security: Exa API key was typed directly in chat. NOT stored in any committed file. Recommend key rotation at https://dashboard.exa.ai/keys.
- Reversal cost: none -- pure documentation, no code change.
- Action: Awaiting user direction. If user wants to act on REPORT.md §9 candidates, will create a Phase 5 plan at docs/plans/anime_sr_landscape_action_plan.md (NOT created yet).

### 2026-09-03 -- WSL2 + Docker probe COMPLETE (unblocks MambaIRv2)
- Decision: Docker Desktop WSL2 backend works end-to-end for MambaIRv2; WSL2 Ubuntu-24.04 native distro does NOT. MambaIRv2 path is VIABLE via Docker (1-2 hour build).
- Method: Probed via pwsh; verified GPU + internet + PyTorch cu124 + Docker build (in progress).
- Key findings (full report at docs/research/anime_sr_2026/WSL2_DOCKER_PROBE.md):
  - WSL2 Ubuntu-24.04 is corporate-locked: no IPv4 on eth0, no default route, no DNS (10.255.255.254 unreachable), curl returns HTTP 000 even with IP. /dev/dxg only (no /dev/nvidia*). apt update fails (no sudo).
  - Docker Desktop WSL2 backend: full internet (pypi HTTP 200, github HTTP 200), GPU passthrough works (nvidia-smi inside container shows RTX 4000 / Driver 595.97 / CUDA 13.2), PyTorch cu124 installs and detects GPU.
  - mamba-ssm + causal-conv1d are source-only on PyPI (no pre-built wheels even for cp311). Must be built from source with nvcc -- need nvidia/cuda:12.4.0-devel-ubuntu22.04 base.
- Dockerfile at tmp/mambair-build/Dockerfile builds: nvidia/cuda:12.4.0-devel-ubuntu22.04 + Python 3.12 + pip (via get-pip.py) + torch cu124 + mamba-ssm + causal-conv1d + basicsr/facexlib/realesrgan + lpips/pyiqa + einops/timm.
- First build attempt failed at apt python3-pip (Py3.12 ships pip 22.0.2 which is broken because distutils was removed in 3.12). Fixed by using get-pip.py from bootstrap.pypa.io.
- Build status: docker build running in background (job pwsh-2); ~30-90 min expected (mostly compilation of mamba-ssm C++/CUDA kernels).
- Updated Phase 5D status: was BLOCKED; now VIABLE via Docker, 1-2 hour build + setup, then training can proceed.
- Reversal cost: none -- tmp/mambair-build/ is .gitignored-able scratch space. The probe report is documentation only.
- Action: Once build completes, run a smoke test inside the container (mamba_ssm imports + 1 epoch of MambaIRv2 inference). If smoke passes, MambaIRv2 is unblocked for Phase 5D.

### 2026-09-03 -- Phase 5 ranked by ROI; Rank #1 = warm-start v1 -> SRVGG no-adv
- Decision: Phase 5 candidates re-sorted by ROI (best quality gain per wall-time). Detailed ranking at docs/research/anime_sr_2026/RECOMMENDED_PATH.md.
- Rank #1 (NEW): Warm-start v1 RFDN -> finetune with SRVGG body + no adversarial + LPIPS weight 1.0. Wall-time: 1-2 days. Reuses I3 code. Expected: PSNR >= 29.0 + lap_var 40-60 + latency 61 ms/frame (1.7x faster than v1).
- Rank #1 is HIGHEST confidence because: (1) reuses all Phase 4 infrastructure, (2) Phase 4 I3 already proved SRVGG body unlocks animevideov3 response signal (in-batch lap_var 3236 at epoch 1 with adv=0), (3) removing --lambda-adv + warm-starting should solve the H6 overshoot that killed I3.
- Rank #2-5 are pure-Python alternatives (no Docker needed). Rank #6 (MambaIRv2) is gated on Docker build completing (~30-60 min remaining). Rank #7 is highest-effort diffusion fine-tune.
- Reversal cost: none -- sorted list is documentation. No code changed.
- Action: pending user direction to pick from rank 1-7. Default recommendation: start Rank #1 as the highest-ROI bet.

### 2026-09-03 -- Phase 5 Rank #1 LAUNCH PREP (warm-start v1 -> SRVGG no-adv)

- Decision: prepare infrastructure for Rank #1 (warm-start v1 RFDN -> TinySRVGGStudent, no adversarial). User chose to keep Docker build (Rank #6 MambaIRv2) running in background and prepare Rank #1 in parallel.
- Method: added --warm-start-mode {strict, partial} flag to distill.py with RFDN -> SRVGG layer mapping; fixed stale CLI recipe in RECOMMENDED_PATH.md (scripts/train.py -> anime_upscaler/distill.py); added wrapper script scripts/run_rank1_warmstart.ps1; added 2 new tests in tests/test_tiny_srvgg.py for partial warm-start.
- Code changes (+~90 LOC total):
  - anime_upscaler/distill.py: argparse --warm-start-mode {strict, partial} (default strict for backward compat); partial path maps RFDN head -> SRVGG body.0 + RFDN upsampler.0 -> SRVGG body.{last}; logs which keys were mapped vs ignored.
  - tests/test_tiny_srvgg.py: 2 new tests (test_partial_warmstart_maps_rfdn_head_to_srvgg_body0 + test_partial_warmstart_forward_works_after_mapping). Total tests: 30 (was 28).
  - scripts/run_rank1_warmstart.ps1 (NEW): wrapper for the full 40-epoch run. Flags: -Smoke, -Epochs, -BatchSize, -Lr, -OutDir, -V1Ckpt, -Python, -SkipWarmStart. Smoke goes to a separate subdir so it doesn't clobber the prod out-dir.
  - docs/research/anime_sr_2026/RECOMMENDED_PATH.md: stale `python scripts/train.py` recipe replaced with verified CLI invocation. Caveat about SRVGG body sharpness bias added (see below).
- Recipe (verified working, smoke 2 ep on RTX 4000, ~80 s wall):
  .venv\\Scripts\\python.exe anime_upscaler\\distill.py --resume pretrained\\RFDN_distill_v1_4x_student.pth --fresh-epoch --warm-start-mode partial --teacher animevideov3 --arch srvgg --lambda-adv 0 --feat-weight 1.0 --shortcut-anneal off --epochs 40 --batch-size 16 --lr 5e-5 --out-dir runs\\distill_v3_4x_srvgg_warmstart
  Or via wrapper: `powershell -ExecutionPolicy Bypass -File scripts\\run_rank1_warmstart.ps1`
- Smoke results (2 epochs, batch 4, 51 + 27 s = 78 s):
  - val PSNR ep 1 = 26.97 dB (vs bicubic 28.55, teacher 29.13); lap_var 3708.7
  - val PSNR ep 2 = 27.01 dB; lap_var 3706.9
  - test PSNR = 27.69 dB (vs bicubic 29.58, teacher 29.12)
- **CAVEAT DISCOVERED (2026-09-03)**: SRVGG body has an inherent sharpness bias that PERSISTS even with warm-start + no adversarial. lap_var >3700 at epoch 1 (no adversarial involved) -- the architecture itself overshoots. This is WORSE than I3's epoch-1 lap_var 3236 (which had adv=0.001 ramp). The Rank #1 hypothesis ('warm-start + no-adv solves H6') may be INCOMPLETE -- the SRVGG body alone triggers overshoot. Full 40-epoch run needed to confirm whether PSNR recovers to >=29.0 or stays below bicubic.
- If PSNR fails to recover, fallback options to try (in order):
  - (a) Add a code change to push LPIPS weight higher (Phase 3 spec currently hardcodes 0.5*L1 + 1.0*LPIPS for SRVGG).
  - (b) Try --warm-start-mode partial with v3 ckpt (warmer start, more conservative).
  - (c) Drop the warm-start idea entirely and rely on --lambda-adv 0 + LPIPS-heavy alone.
  - (d) Move to Rank #2 (APISR-style perceptual) which doesn't carry the SRVGG-bias baggage.
- Reversal cost: low -- new code is opt-in (--warm-start-mode partial); wrapper is independent; new tests are additive.
- Action: Run the full 40-epoch Rank #1 launch when user approves. Default expectation: PSNR 29.0-29.5 dB + lap_var 40-60 + 61 ms/frame; if PSNR stays below bicubic at epoch 20, halt and apply fallback (a).

### 2026-09-03 -- Phase 5 Rank #1 RESULT (warm-start v1 -> SRVGG no-adv) -- NOT PROMOTED

- Result: full 40-epoch run completed in ~17 min wall (40 epochs x ~26 s/epoch).
- Final test set: bicubic 29.45 / **student 27.87** / teacher 29.04 dB.
- H6 triggered: student 27.87 < bicubic 29.45 (1.58 dB below bicubic = overshoot signature).
- D.1 binding gate FAIL: PSNR 27.87 < 29.0 (gap -1.13 dB).
- D.2 SKIPPED (lap_var harness not run on this checkpoint; smoke lap_var 3708+ at epoch 1 suggests D.2 would PASS but is moot given D.1 fail).
- Latency SKIPPED in this run (teacher per-iter 8-14 ms; student latency unchanged from I3 at 61 ms).
- Val trajectory (raw vs EMA): ep1 26.94/25.80 -> ep10 28.14/26.84 -> ep20 28.23/27.56 -> ep30 28.35/27.93 -> ep40 28.39/28.11. Raw plateaued at 28.39 from ep36 onward (improvement < 0.01 dB/epoch in last 5 epochs). EMA consistently 0.25-0.30 dB behind raw (raw overfitting to sharpening).
- **Hypothesis falsified**: warm-start + no-adv does NOT solve H6. SRVGG body has an inherent sharpness bias that persists regardless of warm-start source (v1 RFDN partial mapping) and adversarial weight (0). Test PSNR 27.87 dB matches I3's from-scratch+adv=0.001 result (~27.91 dB) -- warm-start had no measurable effect on the final PSNR.
- Action: per §6 fallback (d), move to Rank #2 (APISR balanced twin perceptual loss).

### 2026-09-05 -- Phase 5 Rank #2 LAUNCH (APISR balanced twin perceptual loss)

- Decision: Rank #1 failed D.1, so pivot to fallback (d) Rank #2. APISR-style balanced twin perceptual loss changes the LOSS family (not just weights) to address SRVGG body sharpness bias by penalising the SR signal in BOTH photoreal (VGG19) and domain-specific (ResNet50) feature spaces -- the architectural-bias hypothesis from Rank #1 suggests this may converge to a less-sharp basin than LPIPS-only.
- Key discovery: TwinPerceptualLoss already exists at src/losses/twin_perceptual_loss.py (320 LOC, implemented 2026-08) and has tests at tests/test_v7_loss_schedule.py::TestDanbooruVGGWeightBalance. We do NOT need to retrain an anime-VGG backbone. The ImageNet-only BSD-3 path is the default when pretrained/danbooru_resnet50.pth is absent (graceful fallback at line 138-139 of twin_perceptual_loss.py). This resolves the licensing uncertainty flagged in §6 "5#2 PREREQUISITE".
- Method: added --loss {vgg, twin} flag to distill.py + --twin-danbooru-weight / --twin-vgg-weight / --twin-delta (legacy alias). Defaults match APISR Ablation Table 4 / v7 roadmap Phase C (0.5/0.5). The flag is opt-in (default --loss vgg preserves Rank #1 behavior); when --loss twin is set under Phase 3 SRVGG spec, the existing LPIPS path is replaced with TwinPerceptualLoss. Non-Phase-3 recipes silently ignore --loss twin (print a one-line warning) to keep the Phase 2 v3 contract stable for any SPAN resumers.
- Code changes (+~110 LOC total):
  - anime_upscaler/distill.py: --loss flag + 3 weight flags + lazy-init _get_twin_perceptual helper (signature-keyed cache, ~40 LOC); loss_gt block dispatches on args.loss (Phase 3 SRVGG spec only); recipe print line includes loss=twin; --twin-delta legacy alias mapped to vgg_weight. Total: +~60 LOC.
  - tests/test_phase5_rank2_twin_perceptual.py (NEW, 8 tests): argparse shape (help text + invalid choice rejection via subprocess), lazy-init signature rebuild cache, frozen-grads guarantee, forward pass on random 32x32 inputs, zero-weight branch (danbooru=1.0, vgg=0.0) finiteness check. Uses subprocess for argparse tests (parser is local to distill.main()). Total tests: 56 (was 48, +8 from Rank #2).
  - scripts/run_rank2_twin_perceptual.ps1 (NEW): wrapper mirroring run_rank1_warmstart.ps1 with --loss twin + twin weight flags. Output dir defaults to runs/distill_v3_4x_srvgg_twin (smoke subdir: runs/distill_v3_4x_srvgg_twin/smoke). Flags: -Smoke, -Epochs, -BatchSize, -Lr, -OutDir, -V1Ckpt, -DanbooruWeight, -VggWeight, -Python, -SkipWarmStart.
- Recipe (verified working, smoke 2 ep on RTX 4000, ~62 s wall):
  .venv\\Scripts\\python.exe anime_upscaler\\distill.py --resume pretrained\\RFDN_distill_v1_4x_student.pth --fresh-epoch --warm-start-mode partial --teacher animevideov3 --arch srvgg --lambda-adv 0 --feat-weight 1.0 --shortcut-anneal off --loss twin --twin-danbooru-weight 0.5 --twin-vgg-weight 0.5 --epochs 40 --batch-size 16 --lr 5e-5 --out-dir runs\\distill_v3_4x_srvgg_twin
  Or via wrapper: `powershell -ExecutionPolicy Bypass -File scripts\\run_rank2_twin_perceptual.ps1`
- Smoke results (2 epochs, batch 4, 37 + 25 s = 62 s):
  - val PSNR ep 1 = 26.95 dB (vs bicubic 28.55, teacher 29.13); ssim_s 0.8407
  - val PSNR ep 2 = 27.01 dB (vs bicubic 28.55, teacher 29.13); ssim_s 0.8582
  - test PSNR = 27.69 dB (vs bicubic 29.58, teacher 29.12)
  - Note: smoke test PSNR matches Rank #1 smoke EXACTLY (27.69 dB); at 2 epochs the loss choice has not yet differentiated. Twin vs LPIPS divergence is expected to manifest over the full 40 epochs.
- Expected wall-time: ~22 min for 40 epochs (vs ~17 min for LPIPS-only; ResNet50 backbone adds ~30% per-iter compute on 256x256 crops).
- Action: full 40-epoch Rank #2 launch in background (pwsh-6). Expected outcome: either (a) PSNR recovers to >=29.0 (twin perceptual provides enough GT-anchor signal to overcome SRVGG body bias -- D.1 PASS, D.2 likely PASS, ship candidate), or (b) PSNR plateaus < 29.0 (architectural bias dominates -- fall to next candidate: try Rank #2 with LPIPS at 0.0 / twin only, or move to Rank #6 MambaIRv2 via Docker rebuild).

### 2026-09-05 -- Phase 5 Rank #2 RESULT (APISR balanced twin perceptual loss) -- NOT PROMOTED, BUT +0.11 dB GAIN OVER RANK #1

- Result: full 40-epoch run completed in 1431 s = ~24 min wall (vs ~17 min for Rank #1).
- Final test set: bicubic 29.45 / **student 27.98** / teacher 29.04 dB.
- H6 triggered: student 27.98 < bicubic 29.45 (1.47 dB below bicubic = overshoot signature still present).
- D.1 binding gate FAIL: PSNR 27.98 < 29.0 (gap -1.02 dB).
- D.2 SKIPPED (lap_var harness not run; smoke lap_var from Rank #1 was 3708+ at epoch 1, suggests D.2 PASSes but is moot given D.1 fail).
- Val trajectory (raw vs EMA): ep1 26.86/25.79 -> ep5 28.12/26.26 -> ep10 28.17/26.82 -> ep15 28.22/27.23 -> ep20 28.31/27.52 -> ep25 28.40/27.75 -> ep30 28.46/27.91 -> ep35 28.47/28.03 -> ep40 28.48/28.14. EMA gap closed from 0.78 dB (ep5) to 0.34 dB (ep40). Raw plateaued at 28.47-28.48 from ep33 onward (improvement < 0.01 dB/epoch in last 8 epochs).
- **Hypothesis partially falsified**: APISR twin perceptual gives +0.11 dB test PSNR vs LPIPS-only Rank #1 (27.98 vs 27.87), and narrows the H6 gap by -0.11 dB (1.47 vs 1.58 dB below bicubic). The SRVGG body sharpness bias is REDUCED but NOT eliminated. Loss family change provides real but bounded improvement.
- Head-to-head vs Rank #1: raw val PSNR consistently +0.05 to +0.10 dB ahead from ep15 onward; EMA converged similarly. Wall-time +7 min (~40% slower) due to ResNet50 backbone compute.
- **Critical finding**: SRVGG body sharpness bias appears STRUCTURAL to the architecture, not specific to the loss function. Both Rank #1 (LPIPS) and Rank #2 (APISR twin) plateau near val 28.48 / test 28.00 dB -- the SRVGG body inductive bias itself drives this ceiling. This is the third independent confirmation (I3 from-scratch 27.91, Rank #1 warm-start 27.87, Rank #2 warm-start+twin 27.98). The remaining options all require architecture change, not loss change.
- Action: per §6 fallback chain, options:
  - (e) Try `--twin-danbooru-weight 1.0 --twin-vgg-weight 0.25` (asymmetric, lean on ResNet50 anime-domain signal alone; cheap ~25 min) -- low expected gain since SRVGG ceiling appears structural
  - (f) Move to Rank #6 (MambaIRv2 via Docker rebuild) -- ~60-90 min Docker rebuild + 2-3 weeks distillation iteration. Different architecture, may break the SRVGG body ceiling.
  - (g) Move to Rank #3 (FRAMER frequency-domain distillation) -- additive loss; may help edge preservation without changing backbone
  - (h) Ship v1 + animevideov3 unchanged per Phase 4 close rule
- Reversal cost: low -- --loss twin is opt-in (default vgg preserves Rank #1 behavior); wrapper is independent; new tests are additive and self-contained.
- Recommendation: option (e) for one more quick data point (~25 min), then (f) MambaIRv2 if still no break -- the SRVGG ceiling is the dominant signal across 3 independent runs and likely needs an architectural change to overcome.

### 2026-09-05 -- Phase 5 Option A LAUNCH (asymmetric twin 1.0/0.25, fallback (e) from 5#2)

- Decision: Run the cheap asymmetric twin ablation. Hypothesis: SRVGG body's structural sharpness bias comes from photo-real VGG19 features pulling toward high-frequency output. Lean 4x harder on anime-domain ResNet50 (drop VGG19 from 0.5 to 0.25, raise ResNet50 from 0.5 to 1.0) to test whether the photo-real anchor is the active ingredient in the +0.11 dB Rank #2 gain. If asymmetric twin converges to < 0.11 dB gain (i.e. the gain was from ResNet50 alone), it confirms the loss-family ceiling is structural to SRVGG body. If it converges to ~similar gain as 5#2, VGG19 is also contributing.
- Method: reuse scripts/run_rank2_twin_perceptual.ps1 with -DanbooruWeight 1.0 -VggWeight 0.25 -OutDir runs\distill_v3_4x_srvgg_twin_asym. NO code changes needed -- the wrapper already accepts both weight flags and any positive values (zero-weight branch already covered by tests/test_phase5_rank2_twin_perceptual.py::test_zero_weight_branch_finite).
- Expected wall-time: ~22-25 min (same ResNet50 backbone compute as 5#2; only the per-iter loss weighting changes).
- Expected outcome: (a) PSNR < 28.0 (asymmetric loses vs balanced -> VGG19 was contributing -> still bounded by SRVGG ceiling); (b) PSNR > 28.05 (asymmetric wins -> ResNet50 alone gives better signal); (c) PSNR ~28.0 (asymmetric same as balanced -> both backbones add similar bounded signal; ceiling is loss-agnostic).
- Action: full 40-epoch launch in background (pwsh-7). Pending result.
- Reversal cost: none -- artifacts in separate out-dir; no code change; recipe is a 2-flag delta from 5#2.

### 2026-09-05 -- Phase 5 Option A RESULT (asymmetric twin 1.0/0.25) -- NOT PROMOTED, FAILS PREDICTION: TIES RANK #1 / REGRESSES -0.11 dB VS RANK #2

- Result: full 40-epoch run completed in ~20 min wall (slightly faster than 5#2's ~24 min).
- Final test set: bicubic 29.45 / **student 27.87** / teacher 29.04 dB.
- H6 triggered: student 27.87 < bicubic 29.45 (1.58 dB below bicubic = overshoot signature). Same magnitude as Rank #1 (also 1.58 dB below bicubic).
- D.1 binding gate FAIL: PSNR 27.87 < 29.0 (gap -1.13 dB).
- D.2 SKIPPED (lap_var harness not run; smoke lap_var from Rank #1 was 3708+ at epoch 1, suggests D.2 PASSes but is moot given D.1 fail).
- Val trajectory (raw vs EMA): ep1 26.97/25.79 -> ep5 28.13/26.27 -> ep10 28.16/26.83 -> ep15 28.21/27.25 -> ep20 28.23/27.56 -> ep25 28.27/27.77 -> ep30 28.33/27.92 -> ep35 28.36/28.02 -> ep40 28.37/28.10. Raw plateaued at 28.37 from ep33 onward (improvement < 0.01 dB/epoch in last 7 epochs). EMA gap closed from 1.18 dB (ep1) to 0.27 dB (ep40).
- **Prediction matrix outcome: FAIL** (VGG19 was contributing). Three-way head-to-head (test PSNR / best val raw / best val EMA / SSIM / H6 gap):
  - Rank #1 (LPIPS-only):           27.87 / 28.39 / 28.11 / 0.8477 / -1.58 dB
  - Rank #2 (balanced twin 0.5/0.5): 27.98 / 28.48 / 28.14 / 0.8465 / -1.47 dB
  - Option A (asymmetric 1.0/0.25):  27.87 / 28.37 / 28.10 / 0.8479 / -1.58 dB
  - **Option A ties Rank #1 and regresses -0.11 dB vs Rank #2**. The +0.11 dB Rank #2 gain was from BALANCED twin regularization, not from anime-domain signal alone.
- **New critical finding**: APISR-style twin perceptual loss works ONLY when VGG19 + ResNet50 are balanced. Unbalancing towards either backbone drops the gain to zero (or below). The balanced twin acts as a multi-view regularizer (photo + anime feature spaces) that pulls the SRVGG body into a less-sharp basin; unbalancing loses that regularization effect.
- **Four-run structural ceiling confirmation**: SRVGG body sharpness bias is STRUCTURAL across FOUR independent runs, all converging near test PSNR 28 dB:
  - I3   (from-scratch + adv=0.001 + LPIPS):           27.91
  - 5#1  (warm-start v1 + LPIPS only):                 27.87
  - 5#2  (warm-start v1 + balanced twin 0.5/0.5):      27.98
  - 5#2b (warm-start v1 + asymmetric twin 1.0/0.25):   27.87
  - Range: 0.11 dB. The loss family is bounded; the architecture is the binding constraint.
- Action: per §6 fallback chain, options:
  - (f) Move to Rank #6 (MambaIRv2 via Docker rebuild) -- ~60-90 min Docker rebuild + 2-3 weeks distillation iteration. Different architecture, may break the SRVGG body ceiling.
  - (g) Move to Rank #3 (FRAMER frequency-domain distillation) -- additive loss; may help edge preservation without changing backbone (~1-2 weeks)
  - (h) Ship v1 + animevideov3 unchanged per Phase 4 close rule (0 min)
- **Recommendation**: (f) MambaIRv2 -- four runs now confirm the SRVGG ceiling; loss-tuning is exhausted. Only an architectural change (MambaIRv2 state-space model) can break it. The Docker build is the bottleneck (~60-90 min) but Dockerfile at tmp/mambair-build/Dockerfile is intact.
- Reversal cost: low -- artifacts in separate out-dir (runs\distill_v3_4x_srvgg_twin_asym); no code change; recipe is a 2-flag delta from 5#2.

### 2026-09-06 -- Phase 5#6 LAUNCH (MambaIRv2 student, pure-PyTorch pivot)

- Decision: User chose to abandon the 60-90 min Docker build (mamba_ssm wheel unbuildable on Windows native; WSL2 path has too many unknowns) and pivot to **pure-PyTorch** MambaIRv2 implementation. The architectural hypothesis (does state-space break the SRVGG-body ceiling?) only requires testing the math, not the CUDA kernel.
- Method: new `anime_upscaler/student_mambair.py` (16 KB): MambaIRv2-style VSSBlock with 4-direction scan (row-fwd, row-rev, col-fwd, col-rev average); closed-form vectorised selective scan via the parallel-prefix trick (cumsum-based, no Python loop); 4-direction intermediate clamp ±20 for numerical safety; 2x-conv head; 8 VSSBlock body; Conv → PixelShuffle → nearest residual tail. `--arch {rfdn,srvgg,mambair}` CLI choice in distill.py (5 surgical edits, all backward-compat default rfdn); `--mambair-embed-dim 48 --mambair-num-blocks 8 --mambair-d-state 16` args. Defaults: 113,952 params.
- First 2 runs OOMed (CUDA OOM at SS2D scan) -- the all-blocks-held-for-backward graph requires ~5 GB on Quadro RTX 4000. Fix: gradient checkpointing on VSSBlock (gated on `self.training`, no eval cost). Reduces training-time peak to 3.6 GB.
- Pilot 10-epoch run, batch=16, animevideov3 teacher, no adversarial, val-batches=4. Started pwsh-6 at 2026-09-06 09:19:32, completed 09:42:31 = **~23 min wall time**. Smoke + standing-up only ~3 min, training+val ~20 min.
- Reversal cost: low -- student_mambair.py is a NEW file (no edit to existing students); distill.py edits are additive (default --arch rfdn preserves Phase 2 v3 / Phase 3 / Phase 4 behavior). All pre-existing tests still pass (28 tests).
- Expected outcome: either (a) Mamba val PSNR climbs to >=29 by ep10 (architectural inductive bias breaks the ceiling -- D.1 PASS, ship candidate pending D.2 + D.3); (b) Mamba tracks SRVGG family curve (27.6-27.9 plateau, no breakthrough; 5#6 fails like prior runs); (c) Mamba blows up to <26 by ep5 (mode-collapse signature from-scratch).
- Decision after pilot data: extend to 30+ epochs if curve is climbing, abort+declare failure if plateau at ep5-6.

### 2026-09-06 -- Phase 5#6 RESULT (pure-PyTorch MambaIRv2 pilot) -- NOT PROMOTED, HYPOTHESIS DISCONFIRMED AT 113K CAPACITY

- Result: full 10-epoch run completed in ~23 min wall.
- **Final test split** (eval via `tmp/eval_mambair_test_psnr.py`, 99 batches, batch=8, 20 s):
  - bicubic 29.53 / **student 26.17** / teacher 29.09 dB
  - student SSIM 0.6289 < bicubic SSIM 0.8796 (student structure WORSE than simple interpolation)
  - student - bicubic = **-3.36 dB** (H6 oversharpening/overfitting halt triggered: student actively HURTS vs no-model baseline)
- **Final full-frame lap_var** (eval via `tmp/eval_mambair_full_frame.py`, 854x480 → 3416x1920, ~18 min due to slow pure-PyTorch scan):
  - bicubic 11.79 / **student 6625.19**
  - **D.2 PASS** trivially (185× threshold), BUT PNG file size ratio student:bicubic = 5.5× (15.4 MB vs 2.8 MB), proving lap_var is from hallucinated HF noise, not real detail. D.2 PASS is meaningless.
- **D.1 binding gate FAIL**: PSNR 26.17 << 29.0 (gap -2.83 dB).
- **D.3 latency FAIL**: pure-PyTorch scan at 1080p = **32,398 ms/frame** (405× over 80 ms budget). CUDA mamba_ssm would be ~50-200× faster (~150-650 ms) but still over budget at 80 ms; only chunked scan + INT8 quantization would fit.
- **Val trajectory** (10 epochs, raw/EMA):
  | Ep | raw | EMA | SSIM | loss_total |
  |---|---|---|---|---|
  | 1 | 26.47 | 25.37 | 0.6338 | 0.463 |
  | 2 | 27.06 | 25.47 | 0.7163 | 0.369 |
  | 3 | 27.46 | 25.59 | 0.7698 | 0.356 |
  | 4 | 27.62 | 25.72 | 0.7873 | 0.342 |
  | 5 | **27.63** (peak) | 25.86 | 0.7857 | 0.335 |
  | 6 | 27.56 (-0.07) | 25.99 | 0.7773 | 0.321 |
  | 7 | 27.47 | 26.13 | 0.7669 | 0.328 |
  | 8 | 27.41 | 26.27 | 0.7599 | 0.324 |
  | 9 | 27.39 | 26.39 | 0.7574 | 0.327 |
  | 10 | **27.37** (-0.26) | 26.51 | 0.7551 | 0.327 |
  - PEAK val raw at ep5 (27.63), REGRESSION through ep10 (27.37). Identical signature to SRVGG family.
  - loss_total descending (0.463 → 0.321 → flat). SSIM peak at ep4 (0.7873).
- **Five-run structural ceiling NOW EXTENDED** (5#6 added):
  - I3   (SRVGG 317K, from-scratch + adv=0.001 + LPIPS):               27.91
  - 5#1  (SRVGG 317K, warm-start v1 + LPIPS only):                     27.87
  - 5#2  (SRVGG 317K, warm-start v1 + balanced twin 0.5/0.5):          27.98
  - 5#2b (SRVGG 317K, warm-start v1 + asymmetric twin 1.0/0.25):       27.87
  - **5#6 (Mamba 113K, from-scratch + no-adversarial, pure-PyTorch):  26.17**  ← WORST
  - range now 0.11 → 1.81 dB across 5 runs. **Mamba 113K UNDER-performs SRVGG family.**
- **Architectural hypothesis DISCONFIRMED at this capacity**: state-space inductive bias (4-direction scan with selective A_log, dt projection, residual skip) did NOT unlock additional PSNR. Mamba's convergence curve tracked SRVGG family (peak ep4-5, plateau + regression thereafter). Three reasons it may under-perform SRVGG:
  1. **Capacity gap (113K vs 317K = 2.8×)**: MambaIRv2 paper baseline is ~2.6M params; 113K is tiny. Even SRVGG at 113K would likely also under-perform SRVGG at 317K.
  2. **Training budget (10 ep vs 40)**: Mamba EMA at 26.51 still climbing (α=0.999 lags); 40 epochs likely gives ~+0.5 dB, putting test ~26.7 dB — STILL below SRVGG family.
  3. **Pure-PyTorch scan inefficiency**: 4-direction closed-form cumsum is fine for forward but may degrade gradient quality (every scan activation contributes non-trivially to backward through cumsum-cumprod chain). CUDA mamba_ssm uses parallel tree reductions that may converge to a less-sharp basin.
- **Decision**: ABANDON pure-PyTorch Mamba path. The 5#6 result is a clean negative result: state-space modeling does not break the structural ceiling at this scale. CUDA mamba_ssm path requires 60+ min Docker build + a much larger Mamba (~2.6M params) with 30+ hours training to be a meaningful test — not worth the commitment without evidence the architecture scales favorably.
- **New pivot**: Phase 5#3 FRAMER frequency-domain distillation (additive loss change, may help edge preservation without changing backbone; ~1-2 weeks effort). The hypothesis there is that frequency-domain constraints preserve low-frequency content (where our gains are) while preventing the high-frequency hallucination (where we lose) — additive loss only, no backbone change, low architectural risk.
- **Files preserved per Q6**:
  - `runs/distill_mambair_v1_10ep_v3/` (10 epoch ckpts + train_log.csv)
  - `runs/distill_mambair_v1_10ep/` (failed dry-run attempt, kept per Q6)
  - `runs/distill_mambair_v1_10ep_v2/` (OOM'd attempt, kept per Q6)
  - `tmp/mambair-build/` (earlier Dockerfile attempt + dry logs, kept for future Docker-based cuda126 mamba_ssm build)
- **Code preserved for future Mamba revival** (if user changes mind on the Docker commitment):
  - `anime_upscaler/student_mambair.py` (NEW, ~16 KB) -- pure-PyTorch MambaIRv2-style student
  - `anime_upscaler/distill.py` (5 surgical edits) -- `--arch mambair` choice + build branch + CLI args
  - `tmp/eval_mambair_test_psnr.py` (NEW, ~140 LOC) -- test-split PSNR/SSIM harness
  - `tmp/eval_mambair_full_frame.py` (NEW, ~110 LOC) -- full-frame lap_var + latency harness
- Reversal cost: low. All changes are additive. No deletions.


---

## 7. Halt conditions / guardrails

### Training halt conditions (apply to any new training run)

- **H1 (collapse)**: Val PSNR < 20 dB at any epoch → mode collapse → halt.
- **H2 (slow)**: Val PSNR < 26 dB after 10 epochs → slow convergence → halt.
- **H3 (plateau)**: EMA PSNR not improving for 10 consecutive epochs → halt.
- **H4 (lap regression)**: Full-frame lap_var < v1 baseline (21.0) → regression → halt.
- **H5 (D2 fail)**: Final lap_var < 35 → D.2 gate fail → don't ship.
- **H6 (oversharpening, NEW 2026-09-03)**: Student test PSNR < bicubic test PSNR → adversarial
  is producing hallucinated edges (high lap_var, low PSNR); halt, document, isolate adversarial.

### Promotion gate (D.1 AND D.2; D.1 binds, NEW 2026-09-03)
- Ship new checkpoint iff full-frame lap_var >= 35 AND PSNR >= baseline AND inference latency <= 200 ms/frame.
- **D.1 (PSNR) is now the binding constraint** (2026-09-03 lesson from I1). A model can trivially
  max lap_var by hallucinating edges; PSNR >= 29.0 dB AND not regressing below v1 is the harder gate.
- Per-epoch ckpts preserved per Q6 rule (do not delete without user confirmation).

### Code guardrails
- No destructive git ops (`--force`, hard reset, branch delete).
- All training writes to `runs/` (gitignored) — never commit `.pth` files.
- Per-epoch rotation must sort by **numeric** epoch (`int(p.stem.split('_')[1])`), not alphabetic (Phase 3 bug `21151e7`).
- LR scheduler step must come AFTER optimizer step (Phase 3 warning).

---

## 8. Cross-references

### Plans and handoffs
- `docs/plans/student_phase4_nearest_adv_srvgg_plan.md` — Phase 4 plan (391 lines).
- `docs/plans/student_v3_result_2026_08.md` — Phase 3 v3 result + lessons.
- `docs/plans/student_adversarial_plan.md` — Phase 3 handoff (~482 lines, 19 sections).
- `docs/plans/student_adversarial_handoff_2026_08.md` — Phase 3 handoff doc (538 lines).
- `docs/rfdn_realtime_report.md` — Phase 1 E2E verification.
- `docs/plans/realtime_4k_plan.md` — Phase 1 master plan.

### Code anchors (use these to find the change site)
- `anime_upscaler/student.py:64-65` — `RFDN.__init__` signature (now includes `shortcut_mode`).
- `anime_upscaler/student.py:75-82` — `RFDN.shortcut_mode` storage + validation.
- `anime_upscaler/student.py:108-118` — forward: conditional `align_corners` for `nearest`/`area` modes.

- `anime_upscaler/student.py:126-132` — `set_shortcut_weight()` (unchanged from Phase 3).
- `anime_upscaler/student.py:134-144` — `set_shortcut_mode()` runtime hook.
- `anime_upscaler/student.py:150-228` — `TinySRVGGStudent` class (Phase 4 I3, 317K params, SRVGGNetCompact-style).
- `anime_upscaler/distill.py:276-278` — `--shortcut-mode {bicubic,nearest}` arg (Phase 4 I1).
- `anime_upscaler/distill.py:279-281` — `--feat-weight {float, default=1.0}` arg (Phase 4 I2).
- `anime_upscaler/distill.py:283-288` — `--arch {rfdn,srvgg}` arg (Phase 4 I3, default rfdn).
- `anime_upscaler/distill.py:381-394` — arch-aware student build (TinySRVGGStudent + forced tap_chans=None when srvgg).
- `anime_upscaler/distill.py:529-532` — student forward dispatch (s_feats=[] when srvgg).
- `anime_upscaler/distill.py:539` — feature distillation gate (`args.feat_weight > 0`).
- `anime_upscaler/distill.py:552` — feature distillation weight multiplier (`args.feat_weight * `).
- `anime_upscaler/distill.py:638-639` — `_shortcut_weight` anneal call (unchanged from Phase 3).
- `anime_upscaler/distill.py:401` — recipe log line now includes `arch=`.
- `tmp/eval_i2a_full_frame.py` — one-off full-frame lap_var eval for I2a/I2b (untracked).
- `tmp/eval_i3_full_frame.py` — one-off full-frame lap_var eval for I3 (untracked; arch-aware).
- `tmp/summarize_i3.py` — per-epoch metric summary for I3 (untracked).
- `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` (vendored RFDN, lines ~560-595) — `shortcut_mode` arg honored at forward (Phase 4 I3 PREREQUISITE met; STALE entry below is RESOLVED).
- `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py` (vendored TinySRVGGStudent, lines ~600-645) — Phase 4 I3 vendored copy, num_params() method.
- `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py::build(kind='srvgg_student')` — Phase 4 I3 dispatch (sniffs num_feat + num_conv + scale from ckpt).
- `apps/anime_upscaler_gui/anime_upscaler_gui/archs.py::build(kind='rfdn_student')` — auto-sniffs shortcut_mode from ckpt args (Phase 4 I3 prerequisite fix).
- `apps/anime_upscaler_gui/anime_upscaler_gui/registry.py::PRESET_CATALOG` — GUI preset catalog (NOT updated for srvgg_student; would only be added on promotion, which didn't happen).
- `tmp/eval_i1_correct_shortcut.py` — one-off full-frame eval with correct shortcut mode (forensics only, untracked).
- `tests/test_shortcut_mode.py` — 4 smoke tests covering default/nearest/runtime-flip/invalid.
- `tests/test_tiny_srvgg.py` — 9 smoke tests covering params/shape/return_features/residual/no-ops/parity/build (Phase 4 I3).

### Eval / harness
- `scripts/compare_students_vs_pretrained.py` — quality harness with `--v3-ckpt` flag.
- `tmp/test_animevideov3_real_video.py` — real-video test for animevideov3.
- `tmp/test_v3_student_real_video.py` — real-video test for v3 student.
- `tmp/test_v1_student_real_video.py` — real-video test for v1 student.
- Test clip: `tmp/real_video_1sec.mp4` (18 frames, 854x480).

---

## 9. Update history

- **2026-09-03**: Phase 5 candidates sorted by ROI; Rank #1 = warm-start v1 -> SRVGG no-adv. New doc docs/research/anime_sr_2026/RECOMMENDED_PATH.md (~9 KB) with full ranking + Rank #1 recipe. REPORT.md section 9 decision matrix re-sorted; WSL2_DOCKER_PROBE.md Alternative section re-sorted; PROJECT_MEMORY section 4 candidates re-numbered 5#1..5#7. Rank #1: warm-start v1 RFDN -> SRVGG body + no adversarial + LPIPS weight 1.0 (1-2 days, reuses I3, expected PSNR >= 29.0 + lap_var 40-60 + latency 61 ms/frame). Rank #6 MambaIRv2 gated on Docker build completion (~30-60 min remaining).

- **2026-09-03**: Phase 5 research landscape survey committed. 15 Exa /search queries (~0.13 USD) covering SOTA, real-time lightweight, Mamba, diffusion, knowledge distillation, cascade, datasets, inference, GAN, edge, TTA, perceptual metrics, hybrid, anime fine-tune, 2025-2026 publications. Decision matrix at REPORT.md §9 lists 9 candidate Phase 5 experiments. Docs at docs/research/anime_sr_2026/ (REPORT.md 22.7 KB / 358 lines, SOURCES.md 51 KB / 447 lines, exa_search_results.json 581 KB raw). No code changes. Security: Exa API key was typed directly in chat -- recommend rotation at https://dashboard.exa.ai/keys.
- **2026-09-03**: Initial creation. Phase 4 plan committed; awaiting approval.
- **2026-09-03**: Memory file created (`docs/PROJECT_MEMORY.md`, 224 lines) + AGENTS.md "Memory protocol — READ FIRST" section added at top of file. Commit `4434609`.
- **2026-09-03**: Self-record creation in §2 HEAD table and §9 update history. Commit `2cb458c`.
- **2026-09-03**: Phase 4.I1 executed. Code added (default backward-compat:
  `shortcut_mode="bicubic"`): `--shortcut-mode` arg, `RFDN(shortcut_mode=...)`,
  `set_shortcut_mode()`, conditional `align_corners` for nearest, 4 new tests
  (`tests/test_shortcut_mode.py`, all passing). 40-epoch run completed
  (~22 min on RTX 4000). **D.2 PASS (lap_var 109.4, 5× v1), D.1 FAIL
  (PSNR 27.97 < v1 29.89 and < bicubic 29.45). NOT PROMOTED.** Result doc:
  `docs/plans/student_phase4_i1_nearest_residual_result.md` (9056 bytes).
  Lesson: sharpness ≠ quality; adversarial + from-scratch + new shortcut =
  oversharpening. Per-epoch ckpts preserved per Q6. Commits `bd72dd3`, `924995f`, `7a7a624`.
- **2026-09-03**: Documentation pass after I1. **§7 new halt condition H6** (oversharpening),
  **§7 promotion gate** now binds on D.1 (PSNR), not just D.2 (lap_var). **§8 code anchors**
  refreshed for new line numbers in `student.py` (shortcut_mode lives at lines 64-65, 75-82,
  108-118, 134-144). AGENTS.md Phase 4 section updated to reflect I1 EXECUTED/NOT
  PROMOTED + decision matrix updated + status snapshot refreshed (HEAD now `7a7a624`).
  Plan file §11 "Approval gate" replaced with execution log. Commit pending below.
- **2026-09-03**: Phase 4.I2 executed (~40 min total: 17 + 24 = ~41 min). Code added to
  `distill.py` (~5 LOC): `--feat-weight {float, default=1.0}` argparse; feature distillation
  block gated by `args.feat_weight > 0` at line 539; weight multiplier at line 552. Two
  warm-start sub-runs from v1: I2a animevideov3+adv+feat=1 (30 ep, 17.3 min, val PSNR EMA
  29.885 = v1, full-frame lap_var **20.8** = v1, held-out test 29.32); I2b SPAN+adv+feat=0
  (40 ep, 23.8 min, val PSNR EMA **29.911** +0.022 over v1, full-frame lap_var **24.2** +15%
  over v1, held-out test 29.35). Both D.2 FAIL, both NOT PROMOTED per matrix. **Three new
  decisions logged** in §6: I2 executed, `feat_weight=0` is the safe default for any future
  SPAN+adv RFDN variant, animevideov3 baseline reaffirmed as the shipping quality option.
  **AGENTS.md** Phase 4 section updated to "post-I2": status header, I2 subsection
  EXECUTED+NOT PROMOTED, decision matrix current state "FAIL FAIL → ship animevideov3
  baseline only", status snapshot. New halt condition remains H6 (oversharpening from I1).
  New result doc: `docs/plans/student_phase4_i2_isolated_adv_result.md` (10196 bytes).
  All 19 pre-existing tests still pass (`test_shortcut_mode` + `test_feature_distillation`).
  Commits: `<this-commit>` (docs pass), `7ac0665` (I2 code + test).
- **2026-09-03**: Phase 4.I3 executed (~21 min wall-time, 40 epochs). Code added ~+320 LOC
  total: new `TinySRVGGStudent` class in `anime_upscaler/student.py` (52 ch × 12 convs + PReLU
  + PixelShuffle + nearest residual, 317,300 params, matches v1's 315K budget); `--arch {rfdn,srvgg}`
  argparse in `distill.py` + arch-aware student build + forward dispatch (s_feats=[] when srvgg);
  vendored `TinySRVGGStudent` + vendored RFDN `shortcut_mode` honor in
  `apps/.../archs.py::build` (sniffs num_feat/num_conv/scale from srvgg_student ckpt;
  auto-sniffs shortcut_mode from rfdn_student ckpt args); new `tests/test_tiny_srvgg.py` with 9
  smoke tests (all passing). Held-out test PSNR **27.91** (D.1 FAIL), val EMA PSNR **28.10**,
  full-frame lap_var **306.48** (D.2 PASS, 5.3× animevideov3, 14.6× v1), latency 61 ms/frame
  (1.7× faster than v1 RFDN). H6 oversharpening halt triggered: student 27.91 < bicubic 29.45.
  Architecture unlock confirmed (lap_var at epoch 1 already 3236 with adv=0); overshooting
  signature matches I1 (from-scratch + new design + adv = oversharpened). All 28 tests pass
  (4 I1 + 15 Phase 2 + 9 I3). **New decisions logged** in §6: I3 executed, SRVGG body
  unlocks animevideov3 signal but overshoots, Phase 4 closed per plan §5 "FAIL FAIL FAIL →
  ship animevideov3 baseline only". **§5 empirical anchors** add I3 row. **§4 pending tasks**
  marks I3 complete (was the only pending item). **§8 code anchors** refreshed for I3 file
  structure. AGENTS.md Phase 4 section will be updated to "post-I3". Result doc:
  `docs/plans/student_phase4_i3_srvgg_body_result.md` (~13 KB). Per Q6, full I3 run
  preserved: `runs/distill_i3_srvgg_body/` with all 40 epochs (last 5 un-archived).

- **2026-09-03**: Phase 5 Rank #1 LAUNCH PREP. Code added (+~90 LOC total): `--warm-start-mode {strict, partial}` arg in `anime_upscaler/distill.py` with RFDN -> SRVGG layer mapping (head -> body.0, upsampler.0 -> body.{last}); wrapper script `scripts/run_rank1_warmstart.ps1`; 2 new tests in `tests/test_tiny_srvgg.py` (partial warm-start mapping + forward sanity). Stale `python scripts/train.py` recipe in `docs/research/anime_sr_2026/RECOMMENDED_PATH.md` replaced with the verified `python anime_upscaler/distill.py` invocation. Smoke test (2 ep, batch 4, 78 s wall): val ep1 PSNR 26.97 dB / lap_var 3708.7; test PSNR 27.69 dB (vs bicubic 29.58). **CAVEAT discovered**: SRVGG body has an inherent sharpness bias that persists even with warm-start + no adversarial -- lap_var >3700 at epoch 1 (no adv involved). The Rank #1 hypothesis may be incomplete; full 40-ep run needed. 30 tests pass (was 28). Docker build pwsh-2 still in background for Rank #6 (MambaIRv2). New decision log entry in §6.
- **2026-09-06**: Phase 5#6 (MambaIRv2 pure-PyTorch) executed + NOT PROMOTED. Code added: `anime_upscaler/student_mambair.py` (NEW, 16 KB) -- pure-PyTorch MambaIRv2-style VSSBlock with 4-direction vectorised selective scan (parallel-prefix trick), LN+dwConv+in_proj+SS2D+out_proj block composition, gamma zero-init residual; grad checkpointing on VSSBlock (gated on `self.training`, eval-passes-through) to prevent OOM during backward; `anime_upscaler/distill.py` 5 surgical edits (--arch choice + runtime guard + build branch + eval branch + 3 CLI args); `tmp/eval_mambair_test_psnr.py` (NEW, ~140 LOC); `tmp/eval_mambair_full_frame.py` (NEW, ~110 LOC). Two OOMs during initial run were caught and fixed by gradient checkpointing on VSSBlock. 10-epoch pilot completed in 23 min wall. Final result: test PSNR **26.17 dB** (D.1 FAIL, -2.83 dB gap), full-frame lap_var **6625** (D.2 PASS-trivially but lap_var is hallucinated HF noise, PNG entropy 5.5× bicubic), H6 triggered (student -3.36 dB below bicubic), D.3 latency **32,398 ms/frame** (pure-PyTorch scan is 405× over 80 ms budget). Five-run structural ceiling now extended to 5#6 -- Mamba 113K UNDER-performs SRVGG family at 317K (26.17 vs 27.87-27.98 dB). **Architectural hypothesis DISCONFIRMED at 113K capacity**: state-space inductive bias did not break the SRVGG-body ceiling; Mamba tracked same regression curve at lower PSNR + 32s/frame inference. Decision: abandon pure-PyTorch Mamba path; CUDA mamba_ssm swap gated on 60+ min Docker build and would only matter if a much-larger Mamba (~2.6M params, 30+ hours train) breaks the ceiling -- not a commitment worth making speculatively. **Pivot to Phase 5#3 (FRAMER frequency-domain distillation, ~1-2 weeks, additive loss change, low architectural risk).** All 28 pre-existing tests still pass. §4 pending tasks marks 5#6 complete + NOT PROMOTED. §3 phase history adds Phase 5#6 subsection. §5 empirical anchors add Mamba row. §5 architecture comparison adds Mamba entry. §6 decision log adds Phase 5#6 LAUNCH + RESULT entries. Commit pending below.

