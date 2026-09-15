"""Input section: queue mode, file picker, current file label, queue listbox, filter.

Real drag-and-drop is enabled when tkinterdnd2 is installed and the root was
created via TkinterDnD.Tk() (see app.py). Drop targets are the panel queue
listbox and the empty-state widget.

If tkinterdnd2 is not installed, the empty state tells the user how to add it,
so we never lie about UI capability.
"""
import tkinter as tk
from tkinter import ttk

from ..ui_constants import DISABLED, FONT_MONO, GROUP_PAD, PAD_X, PAD_Y
from .empty_state import _EmptyState

# tkinterdnd2 is an optional install. If absent, drop targets are not
# registered and the empty state points the user at the pip install command.
try:
    from tkinterdnd2 import DND_FILES  # noqa: F401
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False


# File extensions the drop handler accepts. Match the orchestrator
# accepted types so dropped files do the same thing as the file picker.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
_SUPPORTED_EXTS = _IMAGE_EXTS + _VIDEO_EXTS


class _InputPanel(ttk.LabelFrame):
    def __init__(self, parent, app):
        super().__init__(parent, text="Input", padding=GROUP_PAD)
        self.app = app

        # Mode toggle + buttons
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, PAD_Y))
        ttk.Label(top, text="Mode:").pack(side="left")
        self.queue_mode_var = tk.StringVar(value=app.settings.data.queue_mode)
        ttk.Radiobutton(top, text="Single", variable=self.queue_mode_var,
                        value="single", command=self._on_mode_change).pack(side="left", padx=(PAD_X, PAD_X))
        ttk.Radiobutton(top, text="Batch", variable=self.queue_mode_var,
                        value="batch", command=self._on_mode_change).pack(side="left")
        ttk.Button(top, text="+ Add files", command=app._add_files).pack(side="right")
        ttk.Button(top, text="Clear", command=app._clear_jobs).pack(side="right", padx=(0, PAD_X))

        # Single-mode file label
        self.input_var = tk.StringVar(value="(none)")
        self.input_label = ttk.Label(self, textvariable=self.input_var,
                                     foreground=DISABLED, font=FONT_MONO)
        self.input_label.pack(fill="x", pady=(0, PAD_Y))

        # Batch-mode listbox + empty state (mutually exclusive).
        self.queue_listbox = tk.Listbox(self, height=6, font=FONT_MONO,
                                        selectmode="extended", activestyle="dotbox")
        self.queue_listbox.pack(fill="both", expand=True, pady=(0, PAD_Y))
        # Q2 (perf/queue-controls-gpu-codec): per-job Pause / Cancel /
        # Resume buttons. Disabled by default; enabled by app._refresh_
        # controls() based on the selected job's current status. The
        # Cancel button is enabled for both Running and Paused jobs; the
        # Pause button is enabled only for Running; Resume is enabled
        # only for Paused. Clicking a button pushes a JobControlEvent
        # to the worker via app._send_pause/_send_cancel/_send_resume.
        ctl = ttk.Frame(self)
        ctl.pack(fill="x", pady=(0, PAD_Y))
        self.pause_btn = ttk.Button(ctl, text="❚❚ Pause",
                                     command=lambda: app._send_pause(self._selected_job_id()),
                                     state="disabled")
        self.pause_btn.pack(side="left")
        self.resume_btn = ttk.Button(ctl, text="▶ Resume",
                                      command=lambda: app._send_resume(self._selected_job_id()),
                                      state="disabled")
        self.resume_btn.pack(side="left", padx=(PAD_X, PAD_X))
        self.cancel_btn = ttk.Button(ctl, text="✕ Cancel",
                                      command=lambda: app._send_cancel(self._selected_job_id()),
                                      state="disabled")
        self.cancel_btn.pack(side="left")

        # DnD-accurate empty-state subtitle. The text used to say "drop files
        # here when DnD is enabled" without DnD being wired up -- misleading.
        # Now we tell the truth: DnD-on if tkinterdnd2 is available, else point
        # the user at the pip install that would enable it.
        if _DND_AVAILABLE:
            subtitle = ("Click + Add files or drag-and-drop to upload images or videos.\n"
                        "Tip: File -> Add files (Ctrl+O) also works.")
        else:
            subtitle = ("Click + Add files to upload images or videos.\n"
                        "Tip: File -> Add files (Ctrl+O) also works.\n"
                        "(For drag-and-drop, run: pip install tkinterdnd2-universal, then relaunch.)")
        self._empty = _EmptyState(
            self,
            icon="+",
            title="No files in queue",
            subtitle=subtitle,
            cta_label="Add files",
            cta_command=app._add_files,
        )
        self._empty.pack_forget()

        # Filter + retry/remove
        bot = ttk.Frame(self)
        bot.pack(fill="x")
        ttk.Label(bot, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar(value="All")
        ttk.Combobox(bot, textvariable=self.filter_var,
                     values=["All", "Pending", "Running", "Done", "Error"],
                     state="readonly", width=10).pack(side="left", padx=(PAD_X, PAD_X))
        self.filter_var.trace_add("write", lambda *a: app._refresh_queue_listbox())
        ttk.Button(bot, text="Retry selected", command=app._retry_selected).pack(side="right")
        ttk.Button(bot, text="Remove selected", command=app._remove_selected).pack(side="right", padx=(0, PAD_X))

        self._on_mode_change()
        self._refresh_empty()

        # Bind drag-and-drop if the library is present AND the Tk root
        # supports it (created via TkinterDnD.Tk() -- see app.py).
        # Silently skip otherwise so the GUI works on plain Python installs.
        if _DND_AVAILABLE:
            self._bind_drop_targets()

    # ---- drag and drop ----
    def _bind_drop_targets(self):
        """Wire DND_FILES to the queue listbox + the empty-state widget.

        The drop callback dispatches paths to the orchestrator (app._add_files)
        so the rest of the pipeline sees the same code path as the file picker.
        Wrapped in try because the Tk root might not have been created via
        TkinterDnD.Tk() -- in that case drop_target_register raises.
        """
        try:
            for w in (self.queue_listbox, self._empty):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            # Root is plain tk.Tk() (no DnD). Quietly disable; file picker still works.
            return

    def _on_drop(self, event):
        """Tk drop handler. event.data is a Tcl list of paths.

        We accept only supported image / video extensions and forward to
        app._add_files so the rest of the pipeline sees the same code path
        as the file picker.
        """
        raw = event.data
        paths = self._parse_tcl_list(raw)
        accepted = [p for p in paths if p.lower().endswith(_SUPPORTED_EXTS)]
        if not accepted:
            return
        if callable(getattr(self.app, "_add_files", None)):
            self.app._add_files(paths=accepted)

    @staticmethod
    def _parse_tcl_list(raw: str) -> list:
        """Parse a Tcl-list-style string into a list of plain paths.

        Tcl word-splitting: words are space-separated; brace-protected words
        ({...}) stay intact even with internal spaces. We bracket-split first,
        then strip braces.
        """
        out = []
        i = 0
        while i < len(raw):
            if raw[i] == " ":
                i += 1
                continue
            if raw[i] == "{":
                j = raw.find("}", i + 1)
                if j < 0:
                    out.append(raw[i+1:])
                    break
                out.append(raw[i+1:j])
                i = j + 1
                continue
            j = raw.find(" ", i)
            if j < 0:
                out.append(raw[i:])
                break
            out.append(raw[i:j])
        return out

    # ---- mode toggle ----
    def _on_mode_change(self):
        is_batch = self.queue_mode_var.get() == "batch"
        if is_batch:
            self.input_label.pack_forget()
            self._refresh_empty()
        else:
            self._empty.pack_forget()
            self.queue_listbox.pack_forget()
            self.input_label.pack(fill="x", pady=(0, PAD_Y))

    def _refresh_empty(self):
        """Show empty state iff batch mode and queue is empty."""
        if self.queue_mode_var.get() != "batch":
            return
        if not self.app._jobs:
            self.queue_listbox.pack_forget()
            self._empty.pack(fill="both", expand=True, pady=(0, PAD_Y))
        else:
            self._empty.pack_forget()
            self.queue_listbox.pack(fill="both", expand=True, pady=(0, PAD_Y))

    # ---- Q2 (perf/queue-controls-gpu-codec): per-job control toolbar ----
    def _selected_job_id(self) -> int:
        """Return the job_id of the first selected listbox row, or 0 if none.

        Used by the Pause / Cancel / Resume button commands. Returns 0
        when nothing is selected; the App-level send methods treat 0 as
        a no-op (the worker ignores events for unknown job_ids).
        """
        idxs = self.queue_listbox.curselection()
        if not idxs:
            return 0
        # The listbox rows mirror self.app._jobs 1:1, so index == job index.
        try:
            return self.app._jobs[idxs[0]].id
        except (IndexError, AttributeError):
            return 0

    def _refresh_controls(self):
        """Enable/disable Pause/Resume/Cancel based on the selected job's status.

        Called from app._refresh_queue_listbox() (which runs after every
        listbox mutation or worker event) so the toolbar reflects the
        current state without polling. The buttons are disabled when:

          * no row is selected
          * the selected job is in a terminal state (DONE/ERROR/CANCELLED)
          * the queue mode is single (controls only apply in batch mode)
        """
        from ..state import JobStatus
        # Default: everything disabled.
        pause_state = "disabled"
        resume_state = "disabled"
        cancel_state = "disabled"
        if self.queue_mode_var.get() == "batch":
            idxs = self.queue_listbox.curselection()
            if idxs:
                try:
                    j = self.app._jobs[idxs[0]]
                except (IndexError, AttributeError):
                    j = None
                if j is not None:
                    if j.status == JobStatus.RUNNING:
                        pause_state = "normal"
                        cancel_state = "normal"
                    elif j.status == JobStatus.PAUSED:
                        resume_state = "normal"
                        cancel_state = "normal"
        self.pause_btn.configure(state=pause_state)
        self.resume_btn.configure(state=resume_state)
        self.cancel_btn.configure(state=cancel_state)
