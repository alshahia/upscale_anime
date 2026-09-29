# Fix Plan: Code Review Findings

> Generated from comprehensive review of all subsystems. Organized into 9 work packages (WPs)
> suitable for parallel sub-agent delegation. Each WP is self-contained with clear scope,
> specific fixes, and acceptance criteria.

## Dependency Graph

Phase 1 (parallel):  WP-1  WP-3  WP-4  WP-5  WP-6  WP-7  WP-8  WP-9
                              |
Phase 2 (after WP-1): WP-2 (API - depends on WP-1 security + WP-3 error handling)

**Parallelization:** Launch WP-1, WP-3, WP-4, WP-5, WP-6, WP-7, WP-8, WP-9 simultaneously.
WP-2 starts after WP-1 and WP-3 complete.

---

## WP-1: Security Hardening

**Objective:** Eliminate all security vulnerabilities (command injection, unsafe deserialization, path traversal, missing auth).

**Files:**
- apps/anime_upscaler_gui/anime_upscaler_gui/app_settings_io.py
- src/api/inference_api.py
- src/utils/checkpoint_manager.py
- src/utils/checkpoint_loader.py
- src/inference/engine.py
- src/models/span/neosr_span.py
- src/models/ensemble.py
- src/models/base.py
- anime_upscaler/teacher.py
- anime_upscaler/registry.py
- anime_upscaler/archs.py
- anime_upscaler/distill.py
- anime_upscaler/infer.py
- anime_upscaler/export.py
- anime_upscaler/validate.py
- anime_upscaler/eval_step0.py
- anime_upscaler/_quick_eval_ckpt.py

**Fixes:**

1. **Command injection** (app_settings_io.py:88-90): Replace os.system(f"open '{path}'") / os.system(f"xdg-open '{path}'") with subprocess.run(["open", str(path)]) / subprocess.run(["xdg-open", str(path)]) (list form, no shell).

2. **Unsafe deserialization** (~15 files): Replace the pattern:
   try:
       ckpt = torch.load(path, weights_only=True)
   except UnpicklingError:
       ckpt = torch.load(path, weights_only=False)  # INSECURE
   with the secure opt-in pattern from base_trainer.py:352-372:
   try:
       ckpt = torch.load(path, weights_only=True)
   except Exception:
       if not config.get('security', {}).get('allow_pickle_checkpoint', False):
           raise RuntimeError("Set 'security.allow_pickle_checkpoint: true' to load this checkpoint")
       ckpt = torch.load(path, weights_only=False)
   For files without config access (e.g., infer.py, export.py), add a --allow-pickle CLI flag.

3. **RealESRTeacher unsafe order** (teacher.py:186): Swap to try weights_only=True FIRST, then fall back to False.

4. **API authentication** (inference_api.py): Add API key middleware. Read key from env var DSH_API_KEY. Reject requests with missing/invalid key (401).

5. **API rate limiting** (inference_api.py): Add simple in-memory rate limiter (e.g., 60 req/min per IP). Return 429 when exceeded.

6. **API input size limit** (inference_api.py:329): Add MAX_FILE_SIZE = 50 * 1024 * 1024 (50MB). Check len(image_bytes) before processing. Return 413 if exceeded.

7. **Path traversal** (checkpoint_manager.py:52-57): Add .. to sanitization regex. Reject names containing ..

8. **Download hash verification** (registry.py:445-472): Add sha256 parameter to download(). Embed expected hashes in the preset catalog. Verify after download; delete + raise on mismatch.

**Acceptance criteria:** All security findings resolved. grep -r "os.system" apps/ returns nothing. All torch.load calls use weights_only=True first. API rejects unauthenticated requests.

---

## WP-2: API Repair

**Objective:** Fix the broken /inference endpoint and API robustness issues.

**Depends on:** WP-1 (auth), WP-3 (error handling)

**Files:**
- src/api/inference_api.py
- src/inference/engine.py

**Fixes:**

1. **process_image() missing** (inference_api.py:356): Implement process_image() on InferenceEngine:
   def process_image(self, input_path, output_path, scale=4, enhance_faces=False, tile_size=0):
       img = Image.open(input_path).convert("RGB")
       tensor = self.preprocess(img)
       sr = self.run(tensor) if tile_size <= 0 else self.run_tiled(tensor, tile_size)
       sr_img = self.postprocess(sr)
       sr_img.save(output_path)
       return {"output_path": str(output_path), "scale": scale}
   Or change the API to use engine.run() with proper image I/O.

2. **Validation order** (inference_api.py:311-325): Move scale_factor validation BEFORE the Depends(lambda: get_model(...)) dependency injection. Invalid scale should return 400 without loading a model.

3. **Race conditions** (inference_api.py:64-66,114-118): Add asyncio.Lock around model loading. Use a thread-safe cache with double-checked locking.

4. **Stub endpoint** (inference_api.py:408-424): Either implement inference_get_endpoint or remove it.

**Acceptance criteria:** POST /inference with a valid image returns 200 with upscaled output. Invalid scale returns 400 without loading model. Concurrent requests don't double-load.

---

## WP-3: Error Handling & Monitoring Repair

**Objective:** Fix runtime crashes in error handling and NaN monitoring.

**Files:**
- src/utils/error_handler.py
- src/utils/nan_handler.py

**Fixes:**

1. **logging.handlers not imported** (error_handler.py:8): Add import logging.handlers at the top.

2. **recent_errors key not initialized** (error_handler.py:82-87): Add "recent_errors": [] to the error_stats dict in __init__.

3. **get_error_summary() doesn't exist** (error_handler.py:455): Extract the dead code (lines 380-387) into a proper method:
   def get_error_summary(self):
       return {
           "total_errors": self.error_stats["total_errors"],
           "errors_by_type": self.error_stats["errors_by_type"],
           "errors_by_severity": self.error_stats["errors_by_severity"],
           "recent_errors_count": len(self.error_stats["recent_errors"]),
           "last_error": self.error_stats["recent_errors"][-1] if self.error_stats["recent_errors"] else None
       }

4. **for loop over boolean** (nan_handler.py:508): Change "for self.nan_handler.should_create_backup(self.step):" to "if self.nan_handler.should_create_backup(self.step):"

5. **Duplicate handle_exception** (error_handler.py:270,340): Remove the first definition (line 270).

6. **Dead code** (error_handler.py:206-234,380-387): Remove code after return statements.

**Acceptance criteria:** ErrorHandler() instantiates without error. log_error() works. create_error_summary() works. NaN handler doesn't crash when no NaN detected.

---

## WP-4: Model Correctness

**Objective:** Fix all model architecture bugs that produce wrong results or crash.

**Files:**
- src/models/span/conv_lora.py
- src/models/span/neosr_span.py
- src/models/span/parameter_free_attention.py
- src/models/mamba_pan/hierarchical_mamba.py
- src/models/mamba_pan/mamba_utils.py
- src/models/span/mambair_v2.py
- src/models/teachers/swinir.py
- src/models/ensemble.py

**Fixes:**

1. **ConvLoRA.merge_lora() shape mismatch** (conv_lora.py:108-117): Replace grouped conv with einsum:
   A_matrix = self.lora_A.squeeze(-1).squeeze(-1)  # [r, in]
   B_matrix = self.lora_B.reshape(self.out_channels, self.r, -1)  # [out, r, k*k]
   lora_weight = torch.einsum('or,ri->oir', B_matrix, A_matrix)
   lora_weight = lora_weight.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

2. **NeosrSPAN.forward() self.mean corruption** (neosr_span.py:272): Use local variable:
   if self.is_norm:
       mean = self.mean.type_as(x)
       x = (x - mean) * self.img_range

3. **Spatial attention destroys magnitude** (parameter_free_attention.py:84): Remove sum normalization or use softmax:
   spatial_att = F.softmax(spatial_att.view(B, 1, -1), dim=-1).view(B, 1, H, W)

4. **Fusion weights dead code** (hierarchical_mamba.py:144): Either apply weights before concatenation or remove the parameter.

5. **FallbackMamba SSM missing** (mamba_utils.py:100-106): Implement the selective scan loop or raise NotImplementedError.

6. **SelectiveScanV2 delta = input** (mambair_v2.py:169): Add dt_proj linear layer and use it for delta.

7. **SwinIR missing attention mask** (swinir.py:174-192): Create and pass attention mask for shifted windows.

8. **ConvLoRA main conv not frozen** (conv_lora.py:50-54): Add self.conv.weight.requires_grad_(False).

9. **unmerge_lora() non-functional** (conv_lora.py:129-133): Store original weights before merge, restore in unmerge.

10. **create_ensemble_teacher hardcodes scale=4** (ensemble.py:209,226): Add scale parameter and pass through.

**Acceptance criteria:** All model forward passes produce correct output shapes. LoRA merge/unmerge round-trips. No RuntimeError from shape mismatches.

---

## WP-5: Training Correctness

**Objective:** Fix all training bugs that produce wrong results or break reproducibility.

**Files:**
- src/training/model_a_trainer.py
- src/training/model_b_trainer.py
- src/training/ensemble_trainer.py
- src/training/base_trainer.py
- src/training/orchestrator.py
- src/training/auto_stage_trainer.py
- src/training/stage_controller.py

**Fixes:**

1. **Gradient accumulation broken** (model_a_trainer.py:678): Change to:
   if batch_idx % accumulation_steps == 0:
       optimizer.zero_grad()

2. **EMA model corrupted** (model_b_trainer.py:324-325): After recreating model, add:
   if self.use_ema:
       self.ema_model = self._create_ema_model()

3. **PSNR without clamp** (model_a_trainer.py:1242, model_b_trainer.py:292, ensemble_trainer.py:198): Add pred = pred.clamp(0, 1) before PSNR computation.

4. **No gradient clipping in Model B/Ensemble** (model_b_trainer.py:223, ensemble_trainer.py:137): Add torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) before optimizer.step().

5. **RNG state not saved/restored** (base_trainer.py:280-433): Add to save_checkpoint:
   checkpoint['rng_state'] = {
       'torch': torch.get_rng_state(),
       'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
       'numpy': np.random.get_state(),
       'random': random.getstate(),
   }
   Add to load_checkpoint:
   if 'rng_state' in checkpoint:
       torch.set_rng_state(checkpoint['rng_state']['torch'])
       if torch.cuda.is_available() and checkpoint['rng_state']['cuda']:
           torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
       np.random.set_state(checkpoint['rng_state']['numpy'])
       random.setstate(checkpoint['rng_state']['random'])

6. **Direction-aware FAKD uses identical features** (model_b_trainer.py:197-202): Extract direction-specific features via separate forward passes or model.get_direction_outputs().

7. **EMA doesn't update buffers** (base_trainer.py:198-205): Copy buffers verbatim:
   for ema_buf, model_buf in zip(self.ema_model.buffers(), self.model.buffers()):
       ema_buf.data.copy_(model_buf.data)

8. **Division by zero** (model_b_trainer.py:252, ensemble_trainer.py:164): Add "if num_batches == 0: return {...}" guard.

9. **Ensemble mode blocked** (orchestrator.py:94-98): Remove the NotImplementedError stub; implement ensemble setup or pass through to _train_ensemble.

10. **Duplicate EMA update** (auto_stage_trainer.py:341-342): Remove the redundant call.

11. **Dead code** (model_a_trainer.py:174): Remove unused optimizer creation.

12. **Hardcoded max_norm** (model_a_trainer.py:1091,1095): Read from config: self.config.get('training', {}).get('max_grad_norm', 1.0).

**Acceptance criteria:** Gradient accumulation works with accumulation_steps > 1. EMA produces correct weights. PSNR values are comparable across trainers. Resumed training is reproducible. No division by zero.

---

## WP-6: Data & Augmentation Correctness

**Objective:** Fix all data pipeline bugs that produce wrong training pairs or break reproducibility.

**Files:**
- src/data/augmentation.py
- src/data/quality_analyzer.py
- src/data/base.py
- src/data/preprocessing_manager.py
- src/data/precomputed_dataset.py
- src/data/line_enhancement.py
- src/data/video_dataset.py
- src/data/dataset_sampler.py
- anime_upscaler/dataset.py
- anime_upscaler/degradation.py

**Fixes:**

1. **CutMix axis swap** (augmentation.py:77): Change to:
   hr_mixed[:, cy:cy+cut_h, cx:cx+cut_w] = hr2[:, cy:cy+cut_h, cx:cx+cut_w]

2. **BGR2GRAY on RGB** (quality_analyzer.py:92): Change to cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).

3. **MultiDataset ignores idx** (base.py:1106-1118): Use cumulative length mapping:
   def __getitem__(self, idx):
       for i, ds in enumerate(self.datasets):
           if idx < len(ds):
               return ds[idx]
           idx -= len(ds)
       raise IndexError

4. **MultiDataset __len__ overcounts** (base.py:1104): Return sum(len(d) for d in self.datasets).

5. **OnTheFlyProcessor stub** (preprocessing_manager.py:549-567): Implement full degradation pipeline or raise NotImplementedError.

6. **File handle leaks** (precomputed_dataset.py:93, preprocessing_manager.py:219,294,345,405): Use "with Image.open(f) as img:" context manager.

7. **Small image crash** (anime_upscaler/dataset.py:88): Add guard:
   if w < ch or h < ch:
       hr = hr.resize((max(w, ch), max(h, ch)), Image.BICUBIC)
       w, h = hr.size

8. **Mixup alpha hardcoded** (augmentation.py:190): Use np.random.beta(self.alpha, self.alpha).

9. **Nested Python loops** (line_enhancement.py:123-181): Replace with cv2.connectedComponents and cv2.dilate.

10. **Global RNG side effect** (dataset_sampler.py:38): Use self._rng = np.random.default_rng(seed).

11. **Silent black frame fallback** (video_dataset.py:589): Raise exception or skip with warning.

12. **Color jitter saturation unused** (augmentation.py:240-272): Add saturation adjustment.

13. **"Center crop" does random crop** (base.py:806-814): Either rename or implement actual center crop.

14. **Inconsistent degradation source** (base.py:910-923): Degrade from the same source as HR in both branches.

15. **Hardcoded 256x256 fallback** (preprocessing_manager.py:412-414): Derive from HR image and scale factor.

16. **Crash on small images** (quality_analyzer.py:184): Add "if h < 40 or w < 40: return 0.0" guard.

17. **degradation.py h264/h265 dead code** (degradation.py:70-92): Either add h264/h265 to CODECS or remove the dead code.

**Acceptance criteria:** CutMix produces spatially coherent pairs. Quality metrics use correct color space. MultiDataset is deterministic. No file handle leaks. No crashes on small images.

---

## WP-7: Inference & Export Correctness

**Objective:** Fix inference bugs and export script mismatches.

**Files:**
- src/inference/engine.py
- anime_upscaler/export.py
- anime_upscaler/infer.py
- anime_upscaler/tta.py

**Fixes:**

1. **Device parameter ignored** (engine.py:84): Change to:
   self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

2. **autocast hardcoded to 'cuda'** (engine.py:296): Use self.device.type instead of 'cuda'.

3. **export.py hardcodes RFDN** (export.py:27): Support both RFDN and TinySRVGGStudent. Auto-detect from checkpoint or add --arch flag.

4. **infer.py hardcodes RFDN** (infer.py:32): Same as above.

5. **Mutable default argument** (engine.py:188): Use weights: List[float] = None and create list inside.

6. **run_batch doesn't batch** (engine.py:400-406): Either implement true batching or rename to run_sequential.

7. **Magic number in benchmark** (engine.py:461): Get scale from model or accept as parameter.

8. **export.py calibration reader** (export.py:73): Add guard for w < crop or h < crop.

9. **export.py hardcoded frame index** (export.py:91): Add guard for len(frames) < 8.

**Acceptance criteria:** Device parameter is respected. CPU inference works. export.py and infer.py work with both RFDN and TinySRVGGStudent checkpoints.

---

## WP-8: GUI Robustness

**Objective:** Fix GUI freezes, worker hangs, and frame loss.

**Files:**
- apps/anime_upscaler_gui/anime_upscaler_gui/app.py
- apps/anime_upscaler_gui/anime_upscaler_gui/worker.py
- apps/anime_upscaler_gui/anime_upscaler_gui/decoders.py
- apps/anime_upscaler_gui/anime_upscaler_gui/ffmpeg.py
- apps/anime_upscaler_gui/anime_upscaler_gui/trt_engine.py
- apps/anime_upscaler_gui/anime_upscaler_gui/settings.py
- apps/anime_upscaler_gui/anime_upscaler_gui/backends.py

**Fixes:**

1. **Event loop death** (app.py:851-865): Wrap _handle_event in broad try/except and always reschedule:
   def _poll_events(self):
       try:
           while True:
               evt = self._evt_queue.get_nowait()
               try:
                   self._handle_event(evt)
               except tk.TclError:
                   return
               except Exception as e:
                   log.exception("Error handling event %s", evt)
       except queue.Empty:
           pass
       finally:
           try:
               self.after(100, self._poll_events)
           except tk.TclError:
               pass

2. **Worker blocked 120s** (app.py:704-724): Drain ctl_queue inside the poll loop to check for cancel.

3. **pipe.wait() no timeout** (worker.py:660-663): Use pipe.wait(timeout=30) and handle TimeoutExpired.

4. **_batched() drops partial batch** (worker.py:831-842): After the loop, add:
   if batch:
       yield consumed - len(batch), batch[0] if batch_size == 1 else batch

5. **ZeroDivisionError in download progress** (app.py:774): Guard total > 0 inside _progress.

6. **_cancel_download doesn't cancel** (app.py:811-824): Add threading.Event cancel flag.

7. **_resume_boxes race condition** (app.py:712-723): Use threading.Lock.

8. **_drain_ctl_for_job reorders events** (worker.py:200-218): Drain all, process matching, put back in order.

9. **_Cv2Reader.__new__ hack** (worker.py:497-507): Use factory function.

10. **AsyncReader prefetch hardcoded** (worker.py:508-510): Pass job.prefetch to reader.

11. **print() instead of logging** (decoders.py:266): Use module logger.

12. **AsyncReader producer no cancellation** (decoders.py:256-270): Add threading.Event stop flag.

13. **extract_cut no timeout** (ffmpeg.py:169): Add timeout=60.

14. **NVENC grace period too short** (ffmpeg.py:125-134): Increase to 3-5s.

15. **TRT engine no integrity check** (trt_engine.py:91): Verify file size or checksum.

16. **move_to() doesn't move queue.json** (settings.py:168-184): Move all files or document limitation.

17. **register_backend mutates global** (backends.py:294-295): Use explicit init() function.

18. **APPDATA fallback on Linux** (backends.py:278): Use XDG_CACHE_HOME on Linux.

**Acceptance criteria:** GUI doesn't freeze on worker errors. Cancel works during frame-error modal. No frame loss in batched mode. No worker hangs.

---

## WP-9: Utils & Infrastructure

**Objective:** Fix utility functions, performance optimizer, and metrics.

**Files:**
- src/utils/__init__.py
- src/utils/performance_optimizer.py
- src/utils/metrics.py
- src/utils/loss_stability.py
- src/utils/system.py
- src/utils/pretrained_models.py
- src/inference/__init__.py

**Fixes:**

1. **Stub degradation** (__init__.py:116-161): Instead of replacing with stub_function, raise ImportError with a clear message about which dependency is missing. Or lazy-import with clear error messages.

2. **Buggy condition** (__init__.py:12): Fix operator precedence. Simplify to just "if not _IMPORT_WARNED:"

3. **Buffers converted to half** (performance_optimizer.py:69): Skip BatchNorm buffers:
   if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
       continue

4. **Modules converted to half** (performance_optimizer.py:391): Use torch.autocast instead of converting modules.

5. **_process_in_chunks is a stub** (performance_optimizer.py:341-345): Implement actual chunking or remove.

6. **Deprecated API** (performance_optimizer.py:171): Replace torch.cuda.memory_cached() with torch.cuda.memory_reserved().

7. **SSIM variance negative** (metrics.py:145-147): Add sigma1_sq = sigma1_sq.clamp(min=0).

8. **Loss scaling after clamping** (loss_stability.py:92-93): Apply scaling before clamping.

9. **Division by zero** (system.py:36): Add zero check.

10. **Missing TTA exports** (inference/__init__.py): Add tta_forward and D4_AUGMENTATIONS to __all__.

11. **Hardcoded URLs** (pretrained_models.py:16-33): Consider config file or env var.

**Acceptance criteria:** No silent degradation. Performance optimizer doesn't corrupt BatchNorm. SSIM doesn't produce NaN. No deprecated APIs.

---

## Execution Plan

### Phase 1 (parallel - 8 sub-agents):
Launch all of WP-1, WP-3, WP-4, WP-5, WP-6, WP-7, WP-8, WP-9 simultaneously.

### Phase 2 (after WP-1 + WP-3):
Launch WP-2 (API Repair).

### Phase 3 (integration testing):
After all WPs complete:
1. Run full test suite: python -m pytest tests/ -v
2. Test API end-to-end: start server, send inference request
3. Test GUI: launch, run a job, cancel mid-job
4. Test training: short run with accumulation_steps > 1
5. Test export: export TinySRVGGStudent to ONNX
6. Verify no torch.load(weights_only=False) without opt-in

### Estimated effort:
- WP-1: ~4 hours (security is cross-cutting)
- WP-2: ~2 hours
- WP-3: ~1 hour
- WP-4: ~4 hours (model bugs are subtle)
- WP-5: ~3 hours
- WP-6: ~3 hours
- WP-7: ~2 hours
- WP-8: ~3 hours
- WP-9: ~2 hours

**Total: ~24 hours of sub-agent work, ~6 hours wall-clock with parallelism.**
