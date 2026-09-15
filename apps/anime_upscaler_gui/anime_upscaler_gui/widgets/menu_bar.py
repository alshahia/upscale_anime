"""Menu bar: File / Edit / Run / Tools / Help.

Each entry takes a callback from the orchestrator (`app._method`). We
don't bind to app directly in `__init__` to keep the widget testable
without a full Tk root.
"""
import tkinter as tk
from tkinter import ttk


class _MenuBar(tk.Menu):
    """Top-level menubar. Add via `app.config(menu=menubar)`."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # ---- File ----
        file_m = tk.Menu(self, tearoff=False)
        self.add_cascade(label="File", menu=file_m, underline=0)
        file_m.add_command(label="Add files...", accelerator="Ctrl+O",
                           command=app._add_files)
        file_m.add_command(label="Save settings", accelerator="Ctrl+S",
                           command=app._save_settings)
        file_m.add_separator()
        file_m.add_command(label="Export settings...", command=app._export_settings)
        file_m.add_command(label="Import settings...", command=app._import_settings)
        file_m.add_separator()
        file_m.add_command(label="Quit", accelerator="Ctrl+Q", command=app._on_close)

        # ---- Edit ----
        edit_m = tk.Menu(self, tearoff=False)
        self.add_cascade(label="Edit", menu=edit_m, underline=0)
        edit_m.add_command(label="Clear queue", command=app._clear_jobs)
        edit_m.add_command(label="Retry selected", command=app._retry_selected)
        edit_m.add_command(label="Remove selected", accelerator="Del",
                           command=app._remove_selected)
        edit_m.add_separator()
        edit_m.add_command(label="Reset to defaults", accelerator="Ctrl+R",
                           command=app._reset_settings)

        # ---- Run ----
        run_m = tk.Menu(self, tearoff=False)
        self.add_cascade(label="Run", menu=run_m, underline=0)
        run_m.add_command(label="Start", accelerator="Ctrl+Enter",
                          command=app._on_start)
        run_m.add_command(label="Refresh model list", accelerator="F5",
                          command=app._refresh_model_dropdown)

        # ---- Tools ----
        tools_m = tk.Menu(self, tearoff=False)
        self.add_cascade(label="Tools", menu=tools_m, underline=0)
        tools_m.add_command(label="Open settings folder",
                            command=app._open_settings_folder)
        tools_m.add_command(label="Move app data...",
                            command=app._move_app_data)

        # ---- Help ----
        help_m = tk.Menu(self, tearoff=False)
        self.add_cascade(label="Help", menu=help_m, underline=0)
        help_m.add_command(label="About...", command=app._show_about)
