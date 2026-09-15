"""UpscaleGUI -- the Tk root window.

MVP: single image upscale flow.
Subsequent commits add: batch queue, video cut window, preview pane,
failed-job handling, VRAM guard, tiling, model downloader.

UI structure (Phase 2):
  `_build_ui` composes extracted widgets from `widgets/`:
    - InputPanel      (queue mode, file picker, listbox, filter)
    - ModelPanel      (model dropdown + re-scan)
    - SettingsPanel   (device, precision, scale, batch, decode, prefetch,
                       downscale, GPU guard, tile)
    - CutWindow       (from cut_window.py; hidden by default)
    - OutputPanel     (output mode, folder, suffix, batch folder name)
    - PreviewPane     (from preview.py; fixed 220px)
    - StatusBar       (progress bar + status label)
    - Actions row     (Start / Save / Reset / Open folder) -- stays in app.py
    - ModelsPanel     (listbox, preset dropdown, custom URL, progress)
  The orchestrator owns cross-widget coordination (event handling, save/load,
  download flow, queue mutation).
"""
import logging
import queue


# Phase 5: try to enable real drag-and-drop via tkinterdnd2. Falls back to plain
# tk.Tk() if not installed (the input panel quietly disables DnD binding).
try:
    from tkinterdnd2 import TkinterDnD as _TkinterDnD
    _TK_BASE = _TkinterDnD.Tk
except ImportError:
    _TK_BASE = tk.Tk
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import torch

from .pipeline import PipelineWorker, RunJob, JobEvent
from .registry import ModelRegistry, is_supported_kind
from .settings import AppPaths, QueueController, Settings
from .controllers import (
    compute_output_path as _compute_output_path_ctrl,
    resolve_from_dropdown as _resolve_from_dropdown,
    resolve_kind_from_dropdown as _resolve_kind_from_dropdown,
    resolve_path_from_dropdown as _resolve_path_from_dropdown,
)
from .cut_window import _CutWindow
from .preview import _PreviewPane
from .downloader import ModelDownloader
from .state import Job, JobStatus
from .ui_constants import (ACCENT, BG, DISABLED, ERROR, FG, FONT_BASE, FONT_HEADING,
                            FONT_MONO, OK, PAD_X, PAD_Y, GROUP_PAD, WARN, STATUS_COLORS)
from .theme import apply as _apply_theme, by_name as _theme_by_name
from .errors import humanize as _humanize_error
from .eta import format_eta as _fmt_eta
from .logging_setup import setup_logging, attach_panel
from . import messages as _i18n
from .widgets import (
    _GPUMonitor, _InputPanel, _LogPanel, _LogPanelHandler, _MenuBar, _ModelPanel,
    _OutputPanel, _ModelsPanel, _SettingsPanel, _StatusBar, Toast,
)
from .icons import app_256 as _app_256, load as _load_icon
from .app_settings_io import SettingsIOMixin
from .app_window_chrome import WindowChromeMixin


class UpscaleGUI(SettingsIOMixin, WindowChromeMixin, _TK_BASE):
    def __init__(self):
        super().__init__()
        self.title("Anime Upscaler")
        self.geometry("1200x900")
        self.minsize(800, 600)
        self._set_window_icon()

        # Paths + settings
        self.paths = AppPaths()
        self.settings = Settings(self.paths)
        self.registry = ModelRegistry(self.settings.pretrained_dir_path(),
                                       presets_path=self.paths.registry_path)
        self.queue_ctrl = QueueController(self.paths)

        # If settings were corrupt, queue a one-shot notification after the UI
        # is built (messagebox needs a live root).
        self._settings_corrupt_notice = bool(self.settings.corrupt)

        # Apply the saved theme BEFORE building UI so widget colors are right.
        _apply_theme(_theme_by_name(self.settings.data.theme))

        # Apply saved locale so any side_for()/_(key) calls during _build_ui
        # see the right language. Unknown persisted locale falls back to en.
        _i18n.set_locale(self.settings.data.locale if self.settings.data.locale in _i18n.SUPPORTED_LOCALES else "en")

        # Runtime state
        self.input_path: Optional[Path] = None
        self._job_counter = 0
        self._jobs: List[Job] = self.queue_ctrl.load()
        if self._jobs:
            self._job_counter = max(j.id for j in self._jobs)
        self._in_queue: "queue.Queue" = queue.Queue()
        self._evt_queue: "queue.Queue" = queue.Queue()
        self._worker = PipelineWorker(self._in_queue, self._evt_queue)
        self._worker.start()
        self._running = False
        # Side-channel used by _wait_resume_action (worker thread -> GUI modal).
        # Keyed by job_id so concurrent frame errors don't trample each other.
        self._resume_boxes: dict = {}

        # Initialize logging before the UI so log messages are captured from
        # the first user action onward.
        self.logger = setup_logging(self.paths.logs_dir)
        self._log_handler: Optional[_LogPanelHandler] = None

        self._build_ui()
        self._refresh_model_dropdown()
        self._poll_events()

        if self._settings_corrupt_notice:
            bak = self.paths.settings_path.with_suffix(".bak")
            messagebox.showwarning(
                "Settings file was corrupt",
                f"The settings file at:\n{self.paths.settings_path}\n"
                f"could not be parsed. Defaults have been used; the original was\n"
                f"backed up to:\n{bak}\n\n"
                f"Re-save settings to write a clean copy.",
            )
            self._settings_corrupt_notice = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Menu bar (after _build_ui so app.* widgets exist for callbacks).
        self.menubar = _MenuBar(self, self)
        self.config(menu=self.menubar)
        # Keyboard shortcuts (root-level so they work regardless of focus).
        self._bind_shortcuts()
        # Phase 10: focus outline + status triplet + dynamic wraplength.
        from .a11y import install_focus_outline, status_triplet
        install_focus_outline(self)

    # ============================================================================ #
    # UI construction
    # ============================================================================ #
    def _build_ui(self):
        outer = ttk.Frame(self, padding=GROUP_PAD)
        outer.pack(fill="both", expand=True)

        # Layout (top -> bottom):
        #   Notebook (fills most)
        #     - Upscale tab: Input / Model / Settings / Cut / Output
        #     - Models tab: registry + downloads
        #   Preview pane (fixed 220px)
        #   Status (progress + log)
        #   Actions (Start, Save, Reset, Folder)
        # Actions is guaranteed visible because it has a fixed-height row at the bottom.

        notebook = ttk.Notebook(outer)
        notebook.pack(side="top", fill="both", expand=True, pady=(0, GROUP_PAD))

        upscale_tab = ttk.Frame(notebook)
        notebook.add(upscale_tab, text="Upscale")
        controls_frame, controls_inner = self._make_scrollable(upscale_tab)
        controls_frame.pack(fill="both", expand=True)

        self.input_panel = _InputPanel(controls_inner, self)
        self.input_panel.pack(fill="both", expand=True, pady=(0, GROUP_PAD))

        self.model_panel = _ModelPanel(controls_inner, self)
        self.model_panel.pack(fill="x", pady=(0, GROUP_PAD))

        self.settings_panel = _SettingsPanel(controls_inner, self)
        self.settings_panel.pack(fill="x", pady=(0, GROUP_PAD))

        self.cut_window = _CutWindow(controls_inner)
        self.cut_window.hide()

        self.output_panel = _OutputPanel(controls_inner, self)
        self.output_panel.pack(fill="x", pady=(0, GROUP_PAD))

        models_tab = ttk.Frame(notebook)
        notebook.add(models_tab, text="Models")
        self.models_panel = _ModelsPanel(models_tab, self)
        self.models_panel.pack(fill="both", expand=True)

        # Bottom panels: preview (fixed) + status + actions (always visible).
        bottom = ttk.Frame(outer)
        bottom.pack(side="bottom", fill="x")

        self._build_preview_section(bottom)
        self.status_bar = _StatusBar(bottom, self)
        self.status_bar.pack(fill="both", expand=True)
        self._build_actions_section(bottom)

        # GPU monitor + log panel (sides of the status row; collapse-when-no-GPU)
        self.gpu_monitor = _GPUMonitor(bottom, self)
        self.gpu_monitor.pack(side="left", padx=(GROUP_PAD, 0), pady=(PAD_Y, 0))

        self.log_panel = _LogPanel(bottom, self)
        self.log_panel.pack(side="bottom", fill="x", pady=(PAD_Y, 0))
        # Wire the panel into the logger
        if self._log_handler is None:
            self._log_handler = _LogPanelHandler()
            self._log_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
            ))
            self._log_handler.set_target(self.log_panel)
            attach_panel(self._log_handler)

    def _build_preview_section(self, parent):
        # Fixed-height preview so the actions row underneath is always visible.
        # Use a frame wrapper so we can clamp the height regardless of canvas content.
        wrap = ttk.Frame(parent)
        wrap.pack(fill="x", pady=(0, GROUP_PAD))
        wrap.configure(height=220)
        wrap.pack_propagate(False)
        self.preview_pane = _PreviewPane(wrap)
        self.preview_pane.pack(fill="both", expand=True)

    def _make_scrollable(self, parent):
        """Wrap a vertically-scrollable area: returns (outer_frame, inner_frame)."""
        outer = ttk.Frame(parent)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        # Track inner frame size so the scrollregion updates as content changes.
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Allow mousewheel scrolling when the cursor is over the canvas.
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        # Stretch inner to canvas width.
        def _on_canvas_resize(e):
            canvas.itemconfigure(window_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_resize)
        return outer, inner

    def _build_actions_section(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(PAD_Y, 0))

        def _iconbtn(parent_row, name: str, text: str, command, primary: bool = False):
            ic = _load_icon(name, 24)
            kw = {"text": text}
            if ic is not None:
                kw["image"] = ic
                kw["compound"] = "left"
            btn = ttk.Button(parent_row, command=command, **kw)
            return btn

        self.start_btn = _iconbtn(row, "play", "Start", self._on_start, primary=True)
        self.start_btn.pack(side="left")
        _iconbtn(row, "save", "Save settings", self._save_settings).pack(side="left", padx=(PAD_X, 0))
        _iconbtn(row, "refresh", "Reset to defaults", self._reset_settings).pack(side="left", padx=(PAD_X, 0))
        _iconbtn(row, "save", "Export...", self._export_settings).pack(side="left", padx=(PAD_X, 0))
        _iconbtn(row, "folder-open", "Import...",
                   command=self._import_settings).pack(side="left", padx=(PAD_X, 0))
        ttk.Button(row, text="Move app data...",
                   command=self._move_app_data).pack(side="right", padx=(0, PAD_X))
        ttk.Button(row, text="Open settings folder",
                   command=self._open_settings_folder).pack(side="right")

    # ============================================================================ #
    # Actions
    # ============================================================================ #
    # ---- batch queue ----
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Add images / videos",
            filetypes=[("Images + Video", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.mp4 *.mkv *.avi *.mov *.webm"),
                       ("All files", "*.*")],
        )
        if not paths:
            return
        # In single mode, just keep the first one as input_path.
        if self.input_panel.queue_mode_var.get() == "single":
            self.input_path = Path(paths[0])
            self.input_panel.input_var.set(self.input_path.name)
            self.status_bar.status_var.set(f"Loaded: {self.input_path.name}")
            self._update_cut_window_for_input(self.input_path)
            return
        # In batch mode, add each as a separate job.
        added = 0
        for p in paths:
            pp = Path(p)
            if any(j.input == pp for j in self._jobs):
                continue
            self._jobs.append(Job(
                id=self._next_job_id(),
                input=pp,
                output=self._compute_output_path(pp),
                is_video=self._is_video_path(pp),
            ))
            added += 1
        self._refresh_queue_listbox()
        self.input_panel._refresh_empty()
        self.status_bar.status_var.set(f"Added {added} file(s).")
        if added:
            self._persist_queue()

    def _update_cut_window_for_input(self, path: Path):
        """Show cut window only for videos; probe duration."""
        if self._is_video_path(path):
            self.cut_window.show()
            self.cut_window.probe_async(path)
        else:
            self.cut_window.hide()

    def _compute_output_path(self, input_path: Path) -> Path:
        """Pick an output path based on settings.output_mode + suffix.

        Phase A3: thin wrapper around controllers.job_builder.compute_output_path --
        the orchestrator's only job here is to read settings + widget state and
        pass them as primitives to the pure helper.
        """
        d = self.settings.data
        return _compute_output_path_ctrl(
            input_path,
            output_mode=d.output_mode,
            single_file_suffix=d.single_file_suffix,
            batch_folder_name=d.batch_folder_name,
            custom_output_dir=self.output_panel.output_dir_var.get(),
        )

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def _persist_queue(self) -> None:
        try:
            self.queue_ctrl.persist(self._jobs)
        except Exception as e:
            # Persistence is best-effort; surface a warning but don't block the UI.
            self.status_bar.status_var.set(f"Queue persist failed: {e}")

    def _refresh_queue_listbox(self):
        self.input_panel.queue_listbox.delete(0, "end")
        flt = self.input_panel.filter_var.get()
        from .a11y import status_triplet
        for j in self._jobs:
            if flt != "All" and str(j.status).lower() != flt.lower():
                continue
            glyph, _color, label = status_triplet(j.status)
            line = f"#{j.id:>4}  {glyph} {label:<9}  {j.input.name}"
            if j.status == JobStatus.RUNNING and j.fps:
                line += f"   {j.fps:.2f} fps"
            if j.status == JobStatus.DONE and j.infer_ms:
                line += f"   {j.infer_ms:.1f} ms"
            if j.status == JobStatus.ERROR:
                line += f"   {j.error[:60]}"
            self.input_panel.queue_listbox.insert("end", line)
        # color rows (Phase 10: status triplet — color is one of 3 channels)
        for i, j in enumerate(self._visible_jobs()):
            try:
                self.input_panel.queue_listbox.itemconfig(i, fg=STATUS_COLORS.get(j.status, FG))
            except Exception:
                pass

    def _visible_jobs(self) -> List[Job]:
        flt = self.input_panel.filter_var.get()
        return [j for j in self._jobs if flt == "All" or str(j.status).lower() == flt.lower()]

    def _clear_jobs(self):
        self._jobs.clear()
        self.input_path = None
        self.input_panel.input_var.set("(none)")
        self._refresh_queue_listbox()
        self.input_panel._refresh_empty()
        self._persist_queue()

    def _retry_selected(self):
        idxs = self.input_panel.queue_listbox.curselection()
        if not idxs:
            return
        changed = False
        for i in idxs:
            j = self._visible_jobs()[i]
            if j.status == JobStatus.ERROR:
                j.status = JobStatus.PENDING
                j.error = ""
                changed = True
        self._refresh_queue_listbox()
        if changed:
            self._persist_queue()

    def _remove_selected(self):
        idxs = self.input_panel.queue_listbox.curselection()
        if not idxs:
            return
        for i in reversed(idxs):
            j = self._visible_jobs()[i]
            if j in self._jobs:
                self._jobs.remove(j)
        self._refresh_queue_listbox()
        self.input_panel._refresh_empty()
        self._persist_queue()

    def _pick_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder",
                                       initialdir=self.output_panel.output_dir_var.get())
        if path:
            self.output_panel.output_dir_var.set(path)

    def _refresh_model_dropdown(self):
        self.registry.scan_installed()
        # Put trained (in-house) models first so the user can find their own
        # shipped RFDN at the top of the Upscale tab's model dropdown, with the
        # rest of the installed community weights in their original scan order.
        trained = [m for m in self.registry._installed if m.is_trained]
        others = [m for m in self.registry._installed if not m.is_trained]
        items = []
        for m in trained + others:
            tag = ""
            if m.tainted:
                tag = "  [TAINTED]"
            elif m.is_trained:
                # In-house trained artifact. Visual cue so the user can tell
                # their shipped RFDN apart from downloaded presets in this dropdown
                # (the Models tab dropdown already shows [TRAINED] / [unsupported]).
                tag = "  [TRAINED]"
            elif not m.supported:
                tag = "  [unsupported]"
            elif not is_supported_kind(m.kind or ""):
                tag = "  [no arch]"
            items.append(f"{m.filename}  ({m.kind or '?'}, {m.scale}x, {m.size_mb:.1f} MB){tag}")
        self.model_panel.model_dropdown["values"] = items
        if not items:
            return
        # Pick a default selection. Priority:
        #   1. settings.last_model if it's still installed (substring match,
        #      so it survives filename reformatting).
        #   2. First trained model -- so a fresh launch with a trained artifact
        #      on disk defaults to it (was: defaulted to first alphabetical,
        #      which was 4xLSDIRCompactv2.pth -- confusing when RFDN is shipped).
        #   3. First item (any installed model).
        cur = self.settings.data.last_model
        sel_idx = -1
        if cur:
            for i, it in enumerate(items):
                if cur in it:
                    sel_idx = i
                    break
        if sel_idx < 0 and trained:
            for i, it in enumerate(items):
                if "[TRAINED]" in it:
                    sel_idx = i
                    break
        if sel_idx < 0:
            sel_idx = 0
        self.model_panel.model_dropdown.current(sel_idx)

    def _on_start(self):
        # Resolve which jobs to run.
        if self.input_panel.queue_mode_var.get() == "batch":
            jobs_to_run = [j for j in self._jobs if j.status in (JobStatus.PENDING, JobStatus.ERROR)]
            if not jobs_to_run:
                messagebox.showwarning("Empty queue", "No pending or errored jobs.")
                return
        else:
            if not self.input_path:
                messagebox.showwarning("No input", "Pick an image first.")
                return
            jobs_to_run = None  # signal: synthesize one from input_path below
        sel = self.model_panel.model_dropdown.get()
        if not sel:
            messagebox.showwarning("No model", "Re-scan and pick a model.")
            return
        filename = sel.split()[0]
        chosen = next((m for m in self.registry._installed if m.filename == filename), None)
        if chosen is None:
            messagebox.showerror("Model missing", f"{filename} not in registry.")
            return
        if chosen.tainted:
            if not messagebox.askyesno("Tainted checkpoint",
                f"{filename} has the warm-start taint signature (bias_mean ~0.43).\n"
                "Output will be flat-blue. Continue anyway?"):
                return
        if not chosen.supported or not is_supported_kind(chosen.kind or ""):
            messagebox.showerror("Unsupported arch",
                f"{filename} is kind={chosen.kind!r}; not supported in this MVP.")
            return
        ckpt_abs = str(chosen.path.resolve())

        # Pre-flight: output folder exists + writable, CUDA sanity check.
        if not self._preflight_check(jobs_to_run):
            return

        if jobs_to_run is None:
            # Single mode: synthesize a one-element list with current input_path.
            cut_s, cut_e = self.cut_window.get_seconds()
            jobs_to_run = [Job(
                id=self._next_job_id(),
                input=self.input_path,
                output=self._compute_output_path(self.input_path),
                is_video=self._is_video_path(self.input_path),
                cut_start=cut_s,
                cut_end=cut_e,
            )]
        else:
            # Batch mode: fill in output + model info per job.
            for j in jobs_to_run:
                j.output = self._compute_output_path(j.input)
                j.model_filename = ckpt_abs
                j.kind = chosen.kind
                j.scale = chosen.scale

        self._running = True
        self.start_btn.config(state="disabled")
        self.status_bar.progress_var.set(0.0)
        # Enqueue first job; worker calls back via _job_finished which enqueues the next.
        self._enqueue_next(jobs_to_run)
        self._refresh_queue_listbox()

    def _preflight_check(self, jobs_to_run) -> bool:
        """Verify outputs can be written and CUDA is usable (if requested). Returns False on failure."""
        # 1. Output folder writable
        if jobs_to_run is None:
            outs = [self._compute_output_path(self.input_path)] if self.input_path else []
        else:
            outs = [j.output for j in jobs_to_run]
        for out in outs:
            parent = out.parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
                # Probe writability with a tiny test file
                probe = parent / ".anime_upscaler_write_probe"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError as e:
                messagebox.showerror("Output folder not writable",
                                     f"Cannot write to:\n{parent}\n\n{_humanize_error(e)}")
                return False
        # 2. CUDA sanity (only if the user picked GPU)
        if self.settings_panel.device_var.get() == "cuda" and not torch.cuda.is_available():
            if not messagebox.askyesno(
                "CUDA unavailable",
                "GPU mode selected but torch.cuda.is_available() is False.\n"
                "The job will fall back to CPU (slow).\n\nContinue anyway?",
            ):
                return False
        return True

    def _enqueue_next(self, jobs_to_run):
        """Push the next pending job onto the worker queue."""
        sp = self.settings_panel
        for j in jobs_to_run:
            if j.status in (JobStatus.PENDING, JobStatus.ERROR):
                j.status = JobStatus.RUNNING
                j.error = ""
                j.output.parent.mkdir(parents=True, exist_ok=True)
                is_video = j.is_video or self._is_video_path(j.input)
                # Phase 1.C STATUS: batched video path is implemented in
                # pipeline.py (_process_video_batch) and works correctly for
                # the PyTorch backend, but the TensorRT backend has a stream
                # synchronization bug (NaN/inf after 2-3 iterations when
                # exec_ctx is reused across calls). Until that's resolved,
                # force batch_size=1 for video so the well-tested per-frame
                # path runs. Image jobs can still use the user's batch
                # setting via _run_image's _to_tensor.
                # See docs/plans/realtime_4k_plan.md "Phase 1.C status" for
                # the full diagnosis and the conditions under which batching
                # can be re-enabled (per-frame sync inside _TrtBackend, or
                # per-call exec_ctx rebuild).
                batch_size = 1 if is_video else int(sp.batch_var.get())
                job = RunJob(
                    job_id=j.id,
                    input_path=j.input,
                    output_path=j.output,
                    is_video=is_video,
                    model_filename=j.model_filename or self._resolve_model_path_from_dropdown(),
                    kind=j.kind or self._resolve_kind_from_dropdown(),
                    scale=j.scale or 4,
                    outscale=float(sp.outscale_var.get()),
                    fp16=bool(sp.fp16_var.get()) and sp.device_var.get() == "cuda",
                    device=sp.device_var.get(),
                    batch_size=batch_size,
                    decode=sp.decode_var.get(),
                    prefetch=sp.prefetch_var.get(),
                    pin_memory="auto",
                    downscale_max_edge=int(sp.downscale_var.get()),
                    gpu_guard_mode=sp.gpu_guard_var.get(),
                    tile_size=int(sp.tile_size_var.get()),
                    tile_overlap=int(sp.tile_overlap_var.get()),
                    tta=bool(sp.tta_var.get()),
                    use_tensorrt=bool(sp.use_tensorrt_var.get()),
                    use_nvenc=bool(sp.use_nvenc_var.get()),
                    nvenc_preset=str(sp.nvenc_preset_var.get()),
                    nvenc_qp=int(self.settings.data.nvenc_qp),
                    # Phase 2 (Real-time 4K): auto from model scale (None = auto;
                    # 2x -> cascade 2x2x, 4x -> single shot). Settings doesn't
                    # expose this yet (advanced); power users can edit settings
                    # JSON manually to force a depth.
                    cascade_mode=self.settings.data.cascade_mode,
                    cut_start_seconds=float(j.cut_start),
                    cut_end_seconds=float(j.cut_end),
                    on_frame_error=lambda idx, msg: self._wait_resume_action(j.id, idx, msg),
                )
                self._in_queue.put(job)
                self.status_bar.status_var.set(f"Job #{j.id}: running...")
                return
        # No more jobs.
        self._running = False
        self.start_btn.config(state="normal")
        self.status_bar.status_var.set("All jobs done.")

    @staticmethod
    def _is_video_path(p: Path) -> bool:
        return p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}

    def _ask_frame_error_action(self, job_id: int, frame_idx: int, message: str) -> None:
        """Open a frame-error modal for `job_id` on the GUI thread.

        Non-blocking: returns immediately. The modal stays open until the user
        clicks a button (or closes it). The corresponding entry in
        `self._resume_boxes[job_id]` is what the worker thread is polling;
        the button callback sets `action` and `_done`, then destroys the modal.
        """
        from .pipeline import SKIP_FRAME, SKIP_REST, ABORT_JOB, RETRY_FRAME
        result_box = self._resume_boxes.get(job_id)
        if result_box is None:
            # Race: worker cancelled its box before the GUI scheduled the modal.
            return

        modal = tk.Toplevel(self)
        modal.title(f"Frame error (job #{job_id})")
        modal.transient(self)
        modal.grab_set()
        ttk.Label(modal, text=f"Frame {frame_idx} failed:\n\n{message}",
                  font=FONT_BASE, wraplength=400, justify="left").pack(padx=12, pady=12)

        def _pick(a):
            # Idempotent: clicking twice shouldn't double-write.
            if result_box.get("_done"):
                return
            result_box["action"] = a
            result_box["_done"] = True
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", lambda: _pick(SKIP_FRAME))

        row = ttk.Frame(modal)
        row.pack(padx=12, pady=(0, 12))
        ttk.Button(row, text="Skip frame", command=lambda: _pick(SKIP_FRAME)).pack(side="left", padx=4)
        ttk.Button(row, text="Skip rest of video", command=lambda: _pick(SKIP_REST)).pack(side="left", padx=4)
        ttk.Button(row, text="Abort job", command=lambda: _pick(ABORT_JOB)).pack(side="left", padx=4)
        ttk.Button(row, text="Retry frame", command=lambda: _pick(RETRY_FRAME)).pack(side="left", padx=4)
        # No modal.wait_window() — the GUI thread stays free to process other
        # events while the modal is up. The worker polls result_box["_done"].

    def _wait_resume_action(self, job_id: int, frame_idx: int, message: str) -> str:
        """Worker thread entry point: open a modal on the GUI thread, then poll.

        Each in-flight job gets its own `result_box` so concurrent frame errors
        on different jobs don't trample each other.
        """
        from .pipeline import SKIP_FRAME
        result_box = {"action": SKIP_FRAME, "_done": False}
        self._resume_boxes[job_id] = result_box
        try:
            self.after(0, lambda: self._ask_frame_error_action(job_id, frame_idx, message))
        except Exception:
            self._resume_boxes.pop(job_id, None)
            return SKIP_FRAME
        deadline = time.time() + 120
        while time.time() < deadline:
            if result_box.get("_done"):
                break
            time.sleep(0.05)
        self._resume_boxes.pop(job_id, None)
        return result_box.get("action", SKIP_FRAME)

    def _resolve_model_path_from_dropdown(self) -> str:
        """Phase A3: delegates to controllers.model_resolver."""
        return _resolve_path_from_dropdown(
            self.model_panel.model_dropdown.get(),
            self.registry,
        )

    def _resolve_kind_from_dropdown(self) -> str:
        """Phase A3: delegates to controllers.model_resolver."""
        return _resolve_kind_from_dropdown(
            self.model_panel.model_dropdown.get(),
            self.registry,
        )

    def _download_preset(self):
        sel = self.models_panel.preset_dropdown.get()
        if not sel:
            return
        pid = sel.split()[0]
        p = self.registry.preset_by_id(pid)
        if not p:
            return
        if not is_supported_kind(p.kind):
            messagebox.showwarning("Unsupported", f"{pid} is kind={p.kind!r}; not enabled in this MVP.")
            return
        self._start_download(p.url, p.filename, p.description)

    def _download_custom_url(self):
        url = self.models_panel.custom_url_var.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Paste a URL first.")
            return
        # Best-effort filename from URL path
        from urllib.parse import urlparse
        fn = Path(urlparse(url).path).name or "downloaded.pth"
        self._start_download(url, fn, "custom URL")

    def _start_download(self, url, filename, label):
        import time as _time
        self.models_panel.dl_progress_var.set(0.0)
        self.models_panel.dl_status_var.set(f"Downloading {filename}...")
        self.models_panel.dl_speed_var.set("")
        self.models_panel.dl_eta_var.set("")
        self.models_panel.dl_cancel_btn.pack(fill="x", pady=(PAD_Y, 0))
        self._dl_state = {"start": _time.monotonic(), "last_written": 0, "last_t": _time.monotonic()}
        dl = ModelDownloader(self.registry)

        def _progress(written, total):
            self.models_panel.dl_progress_var.set(written / total * 100.0)
            self.models_panel.dl_status_var.set(
                f"{filename}: {written / 1e6:.1f} / {total / 1e6:.1f} MB"
            )
            # Speed (rolling MB/s over the last sample)
            now = _time.monotonic()
            dt = now - self._dl_state["last_t"]
            if dt >= 0.5:
                dbytes = written - self._dl_state["last_written"]
                if dbytes > 0:
                    mb_s = dbytes / dt / 1e6
                    self.models_panel.dl_speed_var.set(f"{mb_s:.1f} MB/s")
                    remaining = total - written
                    if mb_s > 0:
                        eta_s = remaining / (dbytes / dt)
                        self.models_panel.dl_eta_var.set(f"ETA {_fmt_eta(eta_s)}")
                self._dl_state["last_written"] = written
                self._dl_state["last_t"] = now

        def _done(path):
            self.models_panel.dl_status_var.set(f"Saved -> {path}")
            self.models_panel.dl_speed_var.set("")
            self.models_panel.dl_eta_var.set("")
            self.models_panel.dl_cancel_btn.pack_forget()
            self.models_panel.refresh_listbox()
            self._refresh_model_dropdown()

        def _err(exc):
            self.models_panel.dl_status_var.set(f"FAIL: {exc}")
            self.models_panel.dl_speed_var.set("")
            self.models_panel.dl_eta_var.set("")
            self.models_panel.dl_cancel_btn.pack_forget()
            messagebox.showerror("Download failed", _humanize_error(exc))

        self._dl_thread = dl.download_url(url, filename, on_progress=_progress,
                                          on_done=_done, on_error=_err)

    def _cancel_download(self):
        """Best-effort download cancel. Daemon thread; we just mark the UI as cancelled.

        True cancellation would need a stop signal plumbed into the registry's
        download loop; for now this hides the bar and shows 'Cancelled' so
        the user isn't stuck looking at a progress bar.
        """
        self.models_panel.dl_status_var.set("Cancelled (download may still complete in background)")
        self.models_panel.dl_speed_var.set("")
        self.models_panel.dl_eta_var.set("")
        try:
            self.models_panel.dl_cancel_btn.pack_forget()
        except tk.TclError:
            pass

    def _import_local(self):
        path = filedialog.askopenfilename(
            title="Import .pth/.safetensors into pretrained/",
            filetypes=[("Model files", "*.pth *.safetensors *.pt"), ("All", "*.*")],
        )
        if not path:
            return
        self.models_panel.dl_status_var.set(f"Importing {Path(path).name}...")
        dl = ModelDownloader(self.registry)
        dl.import_local(
            Path(path),
            on_done=lambda m: self._after_import(m),
            on_error=lambda e: (self.models_panel.dl_status_var.set(f"FAIL: {e}"),
                                messagebox.showerror("Import failed", str(e))),
        )

    def _after_import(self, m):
        self.models_panel.dl_status_var.set(f"Imported {m.filename} (kind={m.kind})")
        self.models_panel.refresh_listbox()
        self._refresh_model_dropdown()


    # ============================================================================ #
    # Event polling
    # ============================================================================ #
    def _poll_events(self):
        try:
            while True:
                evt: JobEvent = self._evt_queue.get_nowait()
                try:
                    self._handle_event(evt)
                except tk.TclError:
                    # Window destroyed mid-poll; drop remaining events.
                    return
        except queue.Empty:
            pass
        try:
            self.after(100, self._poll_events)
        except tk.TclError:
            pass

    def _handle_event(self, evt: JobEvent):
        # Find the job record by id
        job_rec = next((j for j in self._jobs if j.id == evt.job_id), None)
        if evt.kind == "started":
            self.status_bar.status_var.set(f"Job #{evt.job_id}: {evt.message}")
            if job_rec:
                job_rec.status = JobStatus.RUNNING
                self._refresh_queue_listbox()
                self._persist_queue()
        elif evt.kind == "progress":
            # progress < 0 means the total frame count is unknown (e.g. PyAV
            # cannot always report stream.frames): show indeterminate instead
            # of a negative percentage.
            indeterminate = evt.progress < 0
            p = max(0.0, min(1.0, evt.progress))
            self.status_bar.progress_var.set(p * 100.0)
            self.status_bar.status_var.set(
                f"Job #{evt.job_id}: "
                + ("(running)" if indeterminate else f"{p * 100:.1f}%")
                + f"  {evt.fps:.2f} fps  {evt.infer_ms:.1f} ms/frame"
                + ("" if evt.frame_idx < 0 else f"  frame {evt.frame_idx}")
            )
            if job_rec:
                job_rec.fps = evt.fps
                job_rec.infer_ms = evt.infer_ms
                self._refresh_queue_listbox()
        elif evt.kind == "finished":
            self.status_bar.progress_var.set(100.0)
            self.status_bar.status_var.set(f"Job #{evt.job_id}: {evt.message}")
            if job_rec:
                job_rec.status = JobStatus.DONE
                self._refresh_queue_listbox()
                self._persist_queue()
            # In batch mode, fire next job if any.
            qmode = self.input_panel.queue_mode_var.get()
            if qmode == "batch" and self._running:
                self._enqueue_next(self._jobs)
                self._refresh_queue_listbox()
            else:
                self.start_btn.config(state="normal")
        elif evt.kind == "error":
            self.status_bar.status_var.set(f"Job #{evt.job_id}: ERROR - {evt.message}")
            if job_rec:
                job_rec.status = JobStatus.ERROR
                job_rec.error = evt.message
                self._refresh_queue_listbox()
                self._persist_queue()
            qmode = self.input_panel.queue_mode_var.get()
            if qmode == "batch" and self._running:
                self._enqueue_next(self._jobs)
                self._refresh_queue_listbox()
            else:
                self.start_btn.config(state="normal")
                messagebox.showerror("Error", f"Job #{evt.job_id}: {evt.message}")
        elif evt.kind == "frame_error":
            # The modal blocks the GUI thread; the on_frame_error callback in the worker
            # is what actually waits. We don't need to do anything here beyond logging.
            pass
        elif evt.kind == "log":
            self.status_bar.status_var.set(f"Job #{evt.job_id}: {evt.message}")
        elif evt.kind == "fatal_error":
            # Worker thread itself died. Surface a banner; re-enable Start so
            # the user can retry (a new worker will be spun up by next Start).
            self.status_bar.status_var.set(f"FATAL: {evt.message}")
            try:
                self.start_btn.config(state="normal")
            except tk.TclError:
                pass
            messagebox.showerror("Worker died", f"The pipeline worker crashed:\n\n{evt.message}")