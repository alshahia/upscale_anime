"""Window/menu ceremony for the GUI (mixin extracted from app.py).

Shortcut bindings, locale toggle, destructive confirmation, about dialog and
the OS window icon. Coordinates anything UI-level into self.* hooks provided
by the orchestrator (UpscaleGUI) -- no state of its own.
"""
from tkinter import messagebox

from . import messages as _i18n


class WindowChromeMixin:
    """Keyboard shortcuts, locale toggle, confirmation, about, window icon."""

    def _bind_shortcuts(self) -> None:
        """Bind global keyboard shortcuts. These work regardless of focus."""
        bindings = {
            "<Control-o>": self._add_files,
            "<Control-s>": self._save_settings,
            "<Control-Return>": self._on_start,
            "<Delete>": self._remove_selected,
            "<F5>": self._refresh_model_dropdown,
            "<Control-q>": self._on_close,
            "<Control-r>": self._reset_settings,
            "<Control-Shift-L>": self._toggle_locale,
        }
        for seq, cb in bindings.items():
            self.bind(seq, lambda e, fn=cb: fn())

    def _toggle_locale(self) -> None:
        """Switch between EN and AR. Save, log, and report which locale became
        active. Rebuilds the UI so rebuilt buttons pick up new languages next
        render. Don't try to live-translate every existing widget -- rebuild,
        same pattern as _on_theme_change.
        """
        next_locale = "ar" if _i18n.active_locale() == "en" else "en"
        _i18n.set_locale(next_locale)
        self.settings.data.locale = next_locale
        self.settings.save()
        self.logger.info("locale switched to %s", next_locale)
        self._build_ui()
        self._refresh_model_dropdown()
        self.models_panel.refresh_listbox()

    def _confirm_destructive(self, title: str, message: str) -> bool:
        """Show a yes/no confirmation for a destructive action.

        Wraps messagebox.askyesno so future enhancements (logging, telemetry,
        do-not-ask-again) live in one place.
        """
        return bool(messagebox.askyesno(title, message))

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About Anime Upscaler GUI",
            "Anime Upscaler GUI\n\n"
            "Tkinter desktop front-end for the upscale_anime toolkit.\n"
            "See docs/onboarding.md for usage and docs/themes.md for theming.",
        )

    def _set_window_icon(self) -> None:
        """Set the OS window/taskbar icon from the duotone asset. No-op if missing."""
        from .icons import load as _il
        for sz in (256, 128, 64, 48, 32):
            ic = _il("app", sz)
            if ic is not None:
                self.iconphoto(True, ic)
                self._app_icon_ref = ic  # keep ref so GC doesn't drop it
                return
