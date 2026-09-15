"""Shared empty-state widget: big icon/letter, title, subtitle, optional CTA.

Use as: pack the empty state where content would go; when the model
list has items, call `.hide()`; otherwise `.show()`.
"""
import tkinter as tk
from tkinter import ttk

from ..ui_constants import BG, FG, TEXT_MUTED, ACCENT, DISABLED, FONT_BASE, FONT_HEADING


class _EmptyState(ttk.Frame):
    """A centered block: big letter/icon, title, subtitle, CTA button."""

    def __init__(self, parent, *, icon: str = "·", title: str = "",
                 subtitle: str = "", cta_label: str = "", cta_command=None,
                 width: int = 60):
        super().__init__(parent)
        self._icon = icon
        self._cta_command = cta_command

        # Big letter
        self._icon_label = tk.Label(self, text=icon, font=("Segoe UI", 36, "bold"),
                                    fg=DISABLED, bg=BG)
        self._icon_label.pack(pady=(8, 4))

        # Title
        self._title_label = tk.Label(self, text=title, font=FONT_HEADING,
                                     fg=FG, bg=BG)
        self._title_label.pack(pady=(0, 2))

        # Subtitle (muted, may wrap)
        self._subtitle_label = tk.Label(self, text=subtitle, font=FONT_BASE,
                                       fg=TEXT_MUTED, bg=BG, wraplength=width * 6,
                                       justify="center")
        self._subtitle_label.pack(pady=(0, 8))

        # CTA button (only if both label and command are provided)
        self._cta_btn = None
        if cta_label and cta_command:
            self._cta_btn = ttk.Button(self, text=cta_label, command=cta_command)
            self._cta_btn.pack(pady=(0, 8))

    def set_title(self, title: str) -> None:
        self._title_label.config(text=title)

    def set_subtitle(self, subtitle: str) -> None:
        self._subtitle_label.config(text=subtitle)

    def set_cta(self, label: str, command) -> None:
        if self._cta_btn is None:
            self._cta_btn = ttk.Button(self, text=label, command=command)
            self._cta_btn.pack(pady=(0, 8))
        else:
            self._cta_btn.config(text=label, command=command)
