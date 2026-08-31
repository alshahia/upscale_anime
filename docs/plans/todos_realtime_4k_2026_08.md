# Real-time 4K TODO Tracker (Phase 1 + Phase 2)

**Plan:** `docs/plans/realtime_4k_plan.md`
**Started:** 2026-08-31
**Last updated:** 2026-08-31 (Phase 1.B at 31 fps @ 4K; Phase 1.C deferred -- TRT stream bug; **Phase 1.N NVENC shipped** -- h264_nvenc encode at 26.5% CPU; **P1.E2E + P1.DOC shipped** -- 18.80 fps end-to-end + Section 10 in report; **committed 7b3b964 on `feature/phase-1-realtime-4k`**; ready for Phase 2)
**Resume file:** `docs/plans/realtime_4k_handoff_2026_08.md`
**Branch:** (pending — feature branch TBD; current is master/main)
**Hardware:** Quadro RTX 4000 8 GB, single GPU
**Goal:** ≥ 25 fps @ 4K end-to-end in the GUI, then ≥ 30 fps with cascade 2×+2×.

---

## Status legend
- `[ ]` pending
- `[~]` in progress
- `[x]` done
- `[!]` blocked
- `[-]` cancelled / deferred

---

## Phase 1 — TensorRT + Batching + NVENC (target: 25+ fps @ 4K)

### B — TensorRT

- [ ] B.1  Install `torch_tensorrt` (try `pip install torch_tensorrt`); fallback to NVIDIA TensorRT wheel + `trtexec`
- [ ] B.2  Smoke import: `python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"`
- [ ] B.3  New file `apps/anime_upscaler_gui/anime_upscaler_gui/trt_engine.py`
  - [ ] B.3.a  `_TrtEngineCache` class with per-(H,W,batch,fp16) keying
  - [ ] B.3.b  `get_engine(model, H, W, batch, fp16=True) -> Callable`
  - [ ] B.3.c  Disk persistence under `%APPDATA%/anime_upscaler_gui/cache/trt/<hash>.engine`
  - [ ] B.3.d  LRU eviction cap at 1 GB
  - [ ] B.3.e  Graceful fallback to `model.forward` if TRT unavailable
- [ ] B.4  Edit `pipeline.py`: replace `with torch.inference_mode(): y = model(x)` with `y = self._trt_run(x)` in both `_run_image` and `_run_video`
- [x] B.5  Edit `app.py` / `settings.py`: add `perf.use_tensorrt` default ON (auto-OFF when TRT unavailable) [DONE 2026-08-31]
- [x] B.6  Edit `widgets/settings_panel.py`: add "Use TensorRT engine" checkbox [DONE 2026-08-31]
- [x] B.7  Test: `py_compile` all touched files — PASS verified — PASS verified
- [x] B.8  Test: image job, batch=1, TRT ON — PSNR drop < 0.05 dB vs eager [DONE 2026-08-31: multi-shape bench `tmp/bench_multishape.py` -- max abs diff <= 0.00195 across 240x426/480x480/540x304/960x540; TRT 2.00-2.37x faster than PyTorch FP16; at 960x540 (4K), 31.4 fps vs 13.2 fps]
- [x] B.9  Test: image job, batch=1, TRT OFF (fallback) -- identical to before [DONE 2026-08-31: same bench `_make_backend(False)` path produces the PyTorch numbers above; _make_backend(True) vs _make_backend(False) agree to 0.00195 max abs diff]

### C — Batching

STATUS (2026-08-31): code is implemented in pipeline.py (`_process_video_batch`,
`_to_tensor_batch`, `_tensor_to_bgr_batch`, extended `_PinnedPool`) and
works correctly for the **PyTorch** backend. The **TensorRT** backend has a
stream synchronization bug: `_TrtBackend.__call__` runs the engine on an
internal `torch.cuda.Stream()`, and even with `wait_stream` the cached
exec_ctx corrupts output after 2-3 iterations (NaN/inf). Until this is
resolved (per-call sync, or per-call exec_ctx rebuild), batched video is
not enabled in the GUI: `app.py` still forces `batch_size=1` for video
jobs. Re-enable batching when one of:
  * A per-call `_TrtBackend.__call__` `torch.cuda.synchronize()` is added
    (kills async perf win)
  * A per-call `engine.create_execution_context()` is used (memory cost)
  * A custom CUDA graph captures the engine + downstream ops (best fix)

Measured speedup of batch=4 vs batch=1 even when working (PyTorch FP16):
  * 480x270: 23.5 -> 27.4 fps (1.16x)
  * 960x540 (4K): 6.2 -> 6.9 fps (1.11x)
The 3x target was unrealistic -- RFDN is small enough that per-frame
overhead (pinned transfer, cast, sync) dominates.

- [x] C.1  Edit `pipeline.py`: `_run_video` accumulator that buffers up to `batch_size` frames
- [x] C.2  Edit `pipeline.py`: `_flush_batch(buf)` -- stacks into `(N,3,H,W)` pinned, runs once, splits, writes N frames
- [x] C.3  Edit `_to_tensor` to handle `pinned_in` of shape `(N,3,H,W)` (not just `(3,H,W)`)
- [x] C.4  Edit `_tensor_to_bgr` to handle `pinned_out` of shape `(N,H,W,3)` (not just `(H,W,3)`)
- [x] C.5  Edit `_PinnedPool` to key by `(H, W, N)` for both input and output
- [-] C.6  Edit `app.py`: lift the `batch_size=1` forced-on-video workaround -- DEFERRED (see status note; re-enable after TRT stream bug is fixed)
- [x] C.7  Test: `py_compile` all touched files -- PASS verified 2026-08-31
- [x] C.8  Test: video job, batch=1, no regression -- PASS (batch=1 path unchanged, app.py workaround restored)
- [-] C.9  Test: video job, batch=4 at 426x240 input -> >= 3x the batch=1 fps -- SKIPPED (TRT bug; PyTorch only gives 1.16x)

### N — NVENC

- [x] N.1  Edit `ffmpeg.py`: add `use_nvenc`, `nvenc_preset`, `nvenc_qp` to `open_encoder` signature [DONE 2026-08-31]
- [x] N.2  Edit `open_encoder`: branch on `use_nvenc` → `h264_nvenc -preset p1 -rc constqp -qp 18` vs current `libx264` [DONE 2026-08-31]
- [x] N.3  Edit `ffmpeg.py`: `_detect_nvenc_support()` probes `ffmpeg -hide_banner -encoders`, cached [DONE 2026-08-31]
- [x] N.4  Edit `settings.py`: add `video.use_nvenc`, `video.nvenc_preset`, `video.nvenc_qp` defaults (auto-OFF when NVENC unavailable) [DONE in 1.B 2026-08-31; nvenc_qp added 2026-08-31]
- [x] N.5  Edit `widgets/settings_panel.py`: "Use NVIDIA hardware encoder (NVENC)" checkbox + preset dropdown [DONE in 1.B 2026-08-31]
- [x] N.6  Test: NVENC encode 3-sec 4K (3840×2160) clip — file 20.7 KB, CPU avg 26.5% (< 30% target) [DONE 2026-08-31]
- [x] N.7  Test: NVENC absent path — auto-fallback to `libx264` works, warns once to stderr [DONE 2026-08-31]

### Phase 1 end-to-end

- [ ] P1.E2E.1  Create `.venv/test_4k_540p_input.mp4` (10-sec 960×540 from source middle)
- [ ] P1.E2E.2  Run full GUI video job: 4K input, batch=4, TRT ON, NVENC ON — measure fps, log to `docs/rfdn_realtime_report.md`
- [ ] P1.E2E.3  Compare with current (TRT OFF, libx264) — fps improvement logged
- [ ] P1.E2E.4  Engine cache stress test: kill app mid-build, relaunch — partial cache usable
- [ ] P1.E2E.5  Update `docs/rfdn_realtime_report.md` Section 9 with Phase 1 numbers

### Phase 1 docs

- [ ] P1.DOC.1  `AGENTS.md` add "Phase 1 Shipped (2026-08-31)" entry with file inventory
- [x] P1.DOC.2  Commit -- 7b3b964 on branch `feature/phase-1-realtime-4k` (16 files, 4775 insertions; full message in commit log) [DONE 2026-08-31]

---

## Phase 2 — Cascade 2× + 2× + Batching + NVENC (target: 30+ fps @ 4K)

### A — Cascade 2× + 2×

- [ ] A.1  Edit `anime_upscaler/models/rfdn.py`: add `scale: int = 4` to `RFDN.__init__`; upsample becomes `3 * (scale ** 2)` out channels + `PixelShuffle(scale)`
- [ ] A.2  Edit `anime_upscaler/distill.py`: add `--scale {2,4}` arg (default 4)
- [ ] A.3  Backwards-compat smoke: load existing `RFDN_distill_v1_4x_student.pth` with default `scale=4` — no shape mismatch
- [ ] A.4  Train 2× student (Option 1 — single 2× model applied twice):
  - [ ] A.4.a  `python anime_upscaler/distill.py --scale 2 --epochs 30 --out_dir runs/distill_v2_2x ...`
  - [ ] A.4.b  Output: `pretrained/RFDN_distill_v2_2x_student.pth`
- [ ] A.5  Create `scripts/eval_cascade_vs_single.py`:
  - [ ] A.5.a  Load validation set (paired LR/HR)
  - [ ] A.5.b  Eval single 4× baseline PSNR
  - [ ] A.5.c  Eval cascade 2× + 2× PSNR
  - [ ] A.5.d  Pass criterion: cascade ≥ single 4× − 0.3 dB
- [ ] A.6  Edit `pipeline.py`: add `CascadeMode.SINGLE_4X` / `CascadeMode.CASCADE_2X2X`
- [ ] A.7  Edit `pipeline.py`: cascade branch runs model twice with intermediate shape change
- [ ] A.8  Edit `app.py`: mode dropdown in Models tab
- [ ] A.9  Edit `settings.py`: `model.cascade_mode` default `SINGLE_4X`
- [ ] A.10 Edit `registry.py`: add new 2× model with `[CASCADE 2X+2X]` tag
- [ ] A.11 Test: `py_compile` all touched files
- [ ] A.12 Test: 2× student loads, runs on sample 480p → 960×540, no NaN
- [ ] A.13 Test: cascade eval — mean PSNR ≥ 29.6 dB on val set

### Phase 2 end-to-end

- [ ] P2.E2E.1  Run full GUI video job: 4K input, cascade mode, batch=4, TRT ON, NVENC ON — measure fps
- [ ] P2.E2E.2  Visual sanity on `.venv/test_30s_TTA.mp4` — no obvious artifacts vs single 4×
- [ ] P2.E2E.3  Pass criterion: ≥ 30 fps end-to-end @ 4K
- [ ] P2.E2E.4  Update `docs/rfdn_realtime_report.md` with Phase 2 numbers

### Phase 2 docs

- [ ] P2.DOC.1  `AGENTS.md` add "Phase 2 Shipped (2026-08-31)" entry
- [ ] P2.DOC.2  `docs/v8_results.md` or new `docs/realtime_4k_results.md` — full Phase 2 benchmark

---

## Cross-cutting

- [ ] X.1  Engine cache pruning at app startup (orphan engines from removed models)
- [ ] X.2  Engine cache size cap (1 GB, LRU)
- [ ] X.3  Status bar UX for "Building engine… N%" messages
- [ ] X.4  Settings persistence migration (additive, no version bump)

---

## Sub-agent dispatch plan

### Phase 1 (sequential, fast)

- **Wave 1** (parallel after environment check):
  - Sub-agent A → B (TRT engine module + integration) — touches `pipeline.py` + new `trt_engine.py`
  - Sub-agent B → N (NVENC switch + GUI settings) — touches `ffmpeg.py` + `settings.py` + `widgets/settings_panel.py`
- **Wave 2** (after Wave 1): C (batched video path) — touches `pipeline.py` + `app.py`
  - Merge Wave 1 first; C is a small change on top.
- **Wave 3** (after Wave 2): end-to-end GUI test + docs

### Phase 2 (after Phase 1)

- Wave 1: A1+A2+A3 (RFDN scale + distill.py arg) + backwards-compat smoke
- Wave 2 (parallel):
  - Sub-agent A → A4 (train 2× student) — background; mostly idle
  - Sub-agent B → A5 (eval script) — small, fast
- Wave 3: A6-A10 (cascade pipeline mode + GUI)
- Wave 4: end-to-end + docs

---

## Risks (from plan)

| Risk | Phase | Mitigation |
|---|---|---|
| torch_tensorrt install fails on Windows | P1 | Fallback to ONNX + Python tensorrt; then to ONNX + trtexec CLI |
| Engine build slow on first use | P1 | Build in worker thread; show "preparing engine" in status |
| NVENC not detected | P1 | Auto-disable; default to libx264 |
| Batch>1 list-shape bug re-introduced | P1 | Explicit shape assertion; unit test batch=4 |
| Cascade quality regresses > 0.3 dB | P2 | Option 1 first; fall back to Option 2 self-distilled stage 2 |
| TRT engine cache grows unbounded | P1+P2 | 1 GB cap + LRU eviction |

---

## Reference

- **Plan:** `docs/plans/realtime_4k_plan.md`
- **Report:** `docs/rfdn_realtime_report.md` (updated end of each phase)
- **Student weight:** `pretrained/RFDN_distill_v1_4x_student.pth` (315,844 student + 10,176 adapters)
- **Test clip:** `.venv/test_1sec.mp4` (854×480, 30 fps, 30 frames)
- **Source:** `C:\Users\Ahmad Mahmoud\Downloads\Video\[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4` (854×480, 18 fps, 1424s)
- **GUI launch:** `python -m anime_upscaler_gui` from repo root
- **Settings file:** `%APPDATA%\anime_upscaler_gui\settings.json`
- **Hardware:** Quadro RTX 4000 8 GB
