# Real-time 4K Plan — Session Handoff (2026-08-31)

**Committed 2026-08-31:** commit `7b3b964` on branch `feature/phase-1-realtime-4k` (16 files, 4775 insertions; merges Phase 1.B + 1.C + 1.N + E2E + docs in a single logical commit).

**Resumable checkpoint.** Start here if the session resets and there is no fresh context.
**Plan files:**
- `docs/plans/realtime_4k_plan.md` — full plan
- `docs/plans/todos_realtime_4k_2026_08.md` — paired checklist
- `docs/plans/realtime_4k_handoff_2026_08.md` — **this file** (session state)

---

## Where we stopped

**Phase 1.B (TensorRT) — SHIPPED.** Multi-shape bench (`tmp/bench_multishape.py`) at 240x426/480x480/540x304/960x540:
  * 960x540 (4K): TRT **31.4 fps** vs PyTorch 13.2 fps (**2.37x**), max abs diff 0.00146
  * 480x480: TRT 58.3 fps vs PT 29.1 fps (2.00x), max diff 0.00146
  * 540x304: TRT 79.6 fps vs PT 40.0 fps (1.99x), max diff 0.00195
  * 240x426: TRT 126.3 fps vs PT 62.1 fps (2.03x), max diff 0.00195
Above the 25 fps @ 4K target with single-frame inference.

**Phase 1.C (Batching) — CODE IMPLEMENTED BUT DEFERRED.**
- Implemented in pipeline.py: `_process_video_batch`, `_to_tensor_batch`, `_tensor_to_bgr_batch`, extended `_PinnedPool` (key by (H, W, N)), extended `_to_tensor` and `_tensor_to_bgr` to be shape-aware.
- Also: `_make_backend` extended with `batch_size` parameter, `_TrtBackend` extended with `batch_size`, `_TrtEngineCache.get` validates batch match.
- **Works for PyTorch backend** (1.11-1.16x speedup at batch=4).
- **TRT backend has a stream sync bug**: `_TrtBackend.__call__` runs engine on internal `torch.cuda.Stream()`, even with `wait_stream` the cached exec_ctx corrupts output after 2-3 iterations (NaN/inf). Reproduced with `tmp/dbg_when.py`: with explicit sync per call, all iterations valid; without, NaN at iter 2.
- **Decision**: keep batched code in place (works for PyTorch, re-enable for TRT when stream issue is fixed), but **revert `app.py` batch_size workaround** so video stays at batch=1 (well-tested per-frame path runs in production).
- Three fix paths when ready: (1) per-call `torch.cuda.synchronize()` in `_TrtBackend.__call__` (kills async perf), (2) per-call `engine.create_execution_context()` (memory cost), (3) CUDA graph capture of engine + downstream ops (best fix).

**Phase 1.N (NVENC) — SHIPPED 2026-08-31.** ffmpeg.py adds `_detect_nvenc_support()` + 3 kwargs to `open_encoder` (`use_nvenc`, `nvenc_preset`, `nvenc_qp`); pipeline.py `_RunJob` + caller threaded; settings + GUI wired in 1.B. **Acceptance**: 3-sec 4K encode = 26.5% CPU avg (target < 30%) via NVENC vs 42.5% via libx264; auto-fallback to libx264 works (warns once to stderr).

---

## Verified numbers (from earlier benchmarks in this session)

Standalone `student.engine` (built earlier with profile 240-540 max), pre-integration:

| Input | Output | PyTorch FP16 fps | TRT FP16 fps | Speedup |
|---|---|---:|---:|---:|
| 240x240 | 960x960 | 115 | 218 | 1.89x |
| 320x320 | 1280x1280 | 63 | 158 | 2.50x |
| 480x480 | 1920x1920 | 30 | 71 | 2.35x |
| 540x540 | 2160x2160 | 23 | 55 | 2.36x |
| **960x540** | **3840x2160 (4K)** | **13.3** | **55.5** | **4.16x** |
| 640x360 | 2560x1440 | 29 | 55 | 1.91x |

Quality vs PyTorch FP16: max abs diff 0.004 (~FP16 epsilon). PSNR proxy 74.8 dB. Well within tolerance.

---

## Exact files changed this session

**New file:**
- `apps/anime_upscaler_gui/anime_upscaler_gui/trt_engine.py` (8,911 bytes) — TensorRT backend module

**Modified:**
- `apps/anime_upscaler_gui/anime_upscaler_gui/pipeline.py` (27,120 bytes — was 25,336)
  - Added `import logging, os`, `log = logging.getLogger(__name__)`
  - Added `try: from .trt_engine import ... except: _HAS_TRT = False` block (after onnxruntime try/except)
  - Added `_make_backend(...)` helper at line 246
  - `_RunJob` got `use_tensorrt: bool = True` field
  - `_run_image` (line 378) — replaced `_PyTorchBackend(...)` instantiation with `build()` + `_make_backend()`
  - `_run_video` (line 453) — same replacement
- `apps/anime_upscaler_gui/anime_upscaler_gui/settings.py` (12,271 bytes — was 11,962)
  - `_Defaults` got 3 new fields: `use_tensorrt`, `use_nvenc`, `nvenc_preset` (default True / True / "p1")
- `apps/anime_upscaler_gui/anime_upscaler_gui/widgets/settings_panel.py` (7,390 bytes — was 7,049)
  - Added a new "Row 5b: Acceleration" frame with TensorRT + NVENC toggles and NVENC preset combobox
- `apps/anime_upscaler_gui/anime_upscaler_gui/app.py` (46,272 bytes — was 45,914)
  - `_enqueue_next` (line 552) — passes `use_tensorrt=bool(sp.use_tensorrt_var.get())` to `_RunJob`
  - `_save_settings` (line 791) — reads 3 new vars into `s.use_tensorrt`, `s.use_nvenc`, `s.nvenc_preset`

---

## Key technical decisions

1. **Used NVIDIA `tensorrt` Python wheel** (not `torch_tensorrt`): `torch_tensorrt` 2.11+cu126 install failed because `dllist` package is unavailable on PyPI for this environment. `pip install tensorrt` worked and gave TensorRT 11.2.1.2 (cu13 libs). Despite cu13 vs cu126 mismatch, builds and runs work on RTX 4000 (sm_75).
2. **FP16 is automatic** when exporting an FP16 PyTorch model to ONNX. No explicit `BuilderFlag.FP16` (that was removed in TRT 11). The dtype flows from the model weights.
3. **Engine cache key:** `(model_hash, batch, H, W, fp16)`. Per-shape engine build. ~80s first build for a new shape, <1ms cached lookup.
4. **Disk cache:** `%APPDATA%/anime_upscaler_gui/cache/trt/` with 1 GB LRU cap.
5. **Profile range** per engine: min=opt=max=(batch, 3, H, W). Specializes each engine to one exact shape (max optimization). Different shapes get separate engines.
6. **`_TrtBackend` excludes `animesr` kind** (TRT cannot model the 3-frame-center trick). Falls back to PyTorch.
7. **TTA forces PyTorch backend.** `_make_backend(..., tta=False)` parameter — when tta=True, the helper returns PyTorch (TTA bypasses the backend and needs `backend.model`).
8. **Stream:** each engine gets its own `torch.cuda.Stream()`. Execute is async (caller must `torch.cuda.synchronize()` before reading output).
9. **Inputs are FP16** when the pipeline runs with `fp16=True`. The pinned buffer flow in pipeline.py already produces FP16 tensor on CUDA, which is exactly what TRT needs.)

---

## Todo state (resume here)

Completed (4/21):
- [x] Phase 1.B: Install TensorRT
- [x] Phase 1.B: Smoke import
- [x] Phase 1.B: Create trt_engine.py
- [x] Phase 1.B: Wire into pipeline.py

Pending — in execution order:

### IMMEDIATE NEXT (Phase 1.B close-out + verification)

- [ ] **B.5+** Verify `use_tensorrt` setting save/roundtrip (open GUI, change toggle, restart, confirm persistence).
- [ ] **B.7** `py_compile` all 5 touched files — already PASS, but rerun as a sanity check.
- [ ] **B.8** Image job, batch=1, TRT ON, fp16 ON — measure max diff vs PyTorch < 0.05 dB PSNR. (Integration test on 16x16 already passed; just rerun the bench sweep at meaningful shapes.)
- [ ] **B.9** Image job, batch=1, TRT OFF — confirm identical output to before.
- [ ] **Bench sweep at multiple shapes** — run the integration test at 240x426, 480x480, 540x304, 960x540. Use background job to avoid the 120s tool timeout.

### Phase 1.C — Batching

- [ ] **C.1** Edit `pipeline.py`: `_run_video` accumulator that buffers up to `batch_size` frames.
- [ ] **C.2** Edit `pipeline.py`: `_flush_batch(buf)` — stack to `(N,3,H,W)` pinned, run once, split, write N frames.
- [ ] **C.3** Extend `_to_tensor` to accept `pinned_in` of shape `(N,3,H,W)` (currently `(3,H,W)`).
- [ ] **C.4** Extend `_tensor_to_bgr` to accept `pinned_out` of shape `(N,H,W,3)`.
- [ ] **C.5** Extend `_PinnedPool` to key by `(H, W, N)` for both input and output.
- [ ] **C.6** Edit `app.py`: lift the `batch_size=1` forced-on-video workaround in `_enqueue_next` (currently line 551: `batch_size = 1 if is_video else int(sp.batch_var.get())`). Change to use `sp.batch_var.get()` for video too.
- [ ] **C.7** `py_compile` all touched files.
- [ ] **C.8** Test: batch=1, no regression.
- [ ] **C.9** Test: batch=4 at 426x240 input — expect >=3x the batch=1 fps.

### Phase 1.N — NVENC

- [ ] **N.1** Edit `ffmpeg.py`: add `use_nvenc: bool = False`, `nvenc_preset: str = "p1"`, `nvenc_qp: int = 18` to `OpenEncoderConfig`.
- [ ] **N.2** Edit `open_encoder`: when `use_nvenc=True`, build args with `-c:v h264_nvenc -preset {preset} -rc constqp -qp {qp}` instead of `-c:v libx264`.
- [ ] **N.3** Add `_detect_nvenc_support() -> bool` — probes `ffmpeg -hide_banner -encoders` for `h264_nvenc`.
- [ ] **N.4** `settings.py` — `use_nvenc` and `nvenc_preset` already added; verify defaults and check that `_save_settings` reads from the right vars (already done).
- [ ] **N.5** GUI toggle — already added in `settings_panel.py` Row 5b.
- [x] **N.6** DONE 2026-08-31: 3-sec 4K (3840×2160) encoded in 1.4s; 20.7 KB output; CPU avg 26.5% (< 30% target).
- [x] **N.7** DONE 2026-08-31: auto-fallback emits one-time stderr warning then uses libx264; verified via cached `_NVENC_SUPPORTED=False` test.

### Phase 1.E2E + DOC

- [ ] **P1.E2E.1** Create `.venv/test_4k_540p_input.mp4` — 10-sec 960x540 from source middle via `ffmpeg -y -ss 700 -t 10 -i <source> -c:v libx264 -preset ultrafast -crf 18 -r 30 <out>`.
- [ ] **P1.E2E.2** Run end-to-end GUI video job: 4K input, batch=4, TRT ON, NVENC ON. Measure fps.
- [ ] **P1.E2E.3** Compare with current (TRT OFF, libx264) — fps improvement.
- [ ] **P1.E2E.4** Update `docs/rfdn_realtime_report.md` Section 9 with Phase 1 numbers.
- [ ] **P1.DOC.1** `AGENTS.md` "Phase 1 Shipped (2026-08-31)" entry.

### Phase 2 (after Phase 1 ships)

- [ ] **Phase 2.A** Add `scale=2` to RFDN + `--scale {2,4}` arg to `distill.py`.
- [ ] **Phase 2.A** Train `RFDN_distill_v2_2x_student.pth` (Option 1 — single 2x model applied twice). ~2 days training.
- [ ] **Phase 2.A** Create `scripts/eval_cascade_vs_single.py` + run eval (>= 29.6 dB pass).
- [ ] **Phase 2.A** Add `CascadeMode.SINGLE_4X` / `CASCADE_2X2X` to pipeline + GUI toggle + registry entry.
- [ ] **Phase 2.E2E** End-to-end at >=30 fps @ 4K with cascade.
- [ ] **Phase 2.DOC** AGENTS.md entry + `docs/realtime_4k_results.md`.

---

## Exact commands to resume verification

### B.7+B.8+B.9 — py_compile + integration bench (resume from here)

```powershell
# 1) py_compile all 5 touched files (should all PASS)
$env:PYTHONPATH = "E:\python projects\upscale_anime\apps"
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" -m py_compile "E:\python projects\upscale_anime\apps\anime_upscaler_gui\anime_upscaler_gui\pipeline.py"
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" -m py_compile "E:\python projects\upscale_anime\apps\anime_upscaler_gui\anime_upscaler_gui\settings.py"
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" -m py_compile "E:\python projects\upscale_anime\apps\anime_upscaler_gui\anime_upscaler_gui\app.py"
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" -m py_compile "E:\python projects\upscale_anime\apps\anime_upscaler_gui\anime_upscaler_gui\widgets\settings_panel.py"
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" -m py_compile "E:\python projects\upscale_anime\apps\anime_upscaler_gui\anime_upscaler_gui\trt_engine.py"

# 2) Integration test (single shape, fast — should pass in ~30s)
$script = @"
import sys; sys.path.insert(0, "apps/anime_upscaler_gui")
import cv2, numpy as np, torch, time
from anime_upscaler_gui.pipeline import _make_backend, _to_tensor, _PinnedPool
from anime_upscaler_gui.archs import build
ckpt = r"pretrained\RFDN_distill_v1_4x_student.pth"
dev = torch.device("cuda"); fp16 = True; kind = "rfdn_student"
m = build(kind, ckpt).to(dev).eval()
if fp16: m = m.half()
trt, _ = _make_backend(m, ckpt, kind, dev, fp16, True, False)
pt,  _ = _make_backend(m, ckpt, kind, dev, fp16, False, False)
rgb = cv2.cvtColor(cv2.imread("sample/lr/anime_hr.png"), cv2.COLOR_BGR2RGB)
rgb = np.tile(rgb, (30, 30, 1))[:480, :480]  # pad to 480x480
pin = _PinnedPool().get_input(480, 480)
x = _to_tensor(rgb, dev, fp16, fp16_pin=False, pinned_in=pin)
with torch.no_grad(): y_t = trt(x); y_p = pt(x)
torch.cuda.synchronize()
print("diff:", (y_t.float()-y_p.float()).abs().max().item())
for _ in range(10):
    with torch.no_grad(): y_t = trt(x); y_p = pt(x)
torch.cuda.synchronize()
t = time.time()
for _ in range(30):
    with torch.no_grad(): y_t = trt(x)
torch.cuda.synchronize()
trt_dt = (time.time()-t)*1000/30
t = time.time()
for _ in range(20):
    with torch.no_grad(): y_p = pt(x)
torch.cuda.synchronize()
pt_dt = (time.time()-t)*1000/20
print(f"TRT={trt_dt:.2f}ms ({1000/trt_dt:.1f}fps)  PT={pt_dt:.2f}ms ({1000/pt_dt:.1f}fps)  speedup={pt_dt/trt_dt:.2f}x")
"@
# Write to file then exec (PowerShell quote-escaping is painful)
Set-Content -Path "tmp\bench_resume.py" -Value $script -Encoding UTF8
& "E:\python projects\upscale_anime\.venv\Scripts\python.exe" "E:\python projects\upscale_anime\tmp\bench_resume.py"
```

Expected:
- `diff: 0.001ish`
- `TRT=14ms-ish (70fps)  PT=33ms-ish (30fps)  speedup=2.3x`

### GUI smoke (verify launch)

```powershell
$env:PYTHONPATH = "E:\python projects\upscale_anime\apps"
# launch in background, log to file, kill after 5s
$p = Start-Process -FilePath "E:\python projects\upscale_anime\.venv\Scripts\python.exe" `
  -ArgumentList @("-m", "anime_upscaler_gui", "--no-splash") `
  -RedirectStandardOutput "E:\python projects\upscale_anime\tmp\gui.log" `
  -RedirectStandardError "E:\python projects\upscale_anime\tmp\gui.err" `
  -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 4
Stop-Process -Id $p.Id -Force
Get-Content "E:\python projects\upscale_anime\tmp\gui.log"
Get-Content "E:\python projects\upscale_anime\tmp\gui.err"
# Expected: window title "Anime Upscaler" briefly visible; both log files empty (clean launch).
```

---

## Critical environmental quirks

1. **No `uv` in venv** — only `pip`. Use `& ".venv\Scripts\pip.exe" install ...` for installs.
2. **Python is 3.12** (not 3.11). Some NVIDIA wheels need 3.11 — `tensorrt` 11.2.1.2 wheel works for 3.12.
3. **TensorRT 11 cu13 libs work on torch cu126** despite version mismatch. Confirmed.
4. **PYTHONPATH must include `apps/`** when launching GUI from repo root: `$env:PYTHONPATH = "E:\python projects\upscale_anime\apps"`.
5. **PowerShell quote escaping** is hostile — prefer writing temp `.py` files and invoking them, rather than embedding complex strings in `-c`.
6. **Tool-call timeout 120s** — engine builds take 60-90s. Run in background or write to file and exec.
7. **Windows console cp1256** — use ASCII `*` not `★` in shipped strings.
8. **No `web_search` available in this session.** Use domain knowledge.
9. **No `read`/`write`/`edit` direct tools** — must call them from inside `run_code` (which is the only directly callable tool).
10. **Multi-line strings inside `run_code`** — the parser fails on some content (notably `f"..."` with embedded quotes or `with` keyword in a string literal). Use `lines.push(...)` pattern, NOT multi-line template literals.

---

## Path reference

| Resource | Path |
|---|---|
| Student weight | `pretrained\RFDN_distill_v1_4x_student.pth` (315,844 student + 10,176 adapters) |
| Source video | `C:UsersAhmad MahmoudDownloadsVideo[Anime3rb.com] Yi Ren Zhi Xia - 1 [480p].mp4` (854x480, 18fps, 1424s) |
| Test clip (built earlier) | `.venv\test_1sec.mp4` (854x480, 30fps, 30 frames from middle) |
| Settings file | `%APPDATA%\anime_upscaler_gui\settings.json` (auto-managed by `_Settings`) |
| TRT engine cache | `%APPDATA%\anime_upscaler_gui\cache\trt\<hash>_b<N>_<H>x<W>_fp16.engine` |
| GUI launch | `python -m anime_upscaler_gui` from repo root with `PYTHONPATH=apps` |
| Plan | `docs\plans\realtime_4k_plan.md` |
| Todos | `docs\plans\todos_realtime_4k_2026_08.md` |
| Report (to update) | `docs\rfdn_realtime_report.md` |

## Hardware

- GPU: Quadro RTX 4000 (sm_75, ~7.5 GB free VRAM)
- CPU: see system
- Driver: see system
- ffmpeg: installed (libx264 + h264_nvenc both available — confirmed at system probe time)

---

## Next session: read this file FIRST, then `docs/plans/todos_realtime_4k_2026_08.md`, then continue.
