"""User-facing strings.

Phase 14: English + Arabic translation tables, locale switcher.
Ponytail: stdlib only. Babel not in requirements; gettext's compile step
adds friction without proportional value when the table fits in two dicts.

Active locale is a module-level mutable. `set_locale(name)` swaps it;
`_(key, **kwargs)` formats the active catalog. Unknown keys fall back to
the English string so a missing translation never produces English-only
gibberish that's worse than the English source.
"""
from __future__ import annotations

from typing import Callable, Dict


SUPPORTED_LOCALES = ("en", "ar")
DEFAULT_LOCALE = "en"


_EN: Dict[str, str] = {
    "tainted_checkpoint_title": "Tainted checkpoint",
    "tainted_checkpoint_message":
        "This checkpoint looks like a warm-start taint (upsampler bias in [0.35, 0.45]).\n"
        "Outputs may be flat. Continue anyway?",
    "unsupported_kind": "{filename} is kind={kind!r}; not supported in this MVP.",
    "unsupported_preset": "{pid} is kind={kind!r}; not enabled in this MVP.",
    "no_url": "Paste a URL first.",
    "download_failed": "Download failed",
    "import_failed": "Import failed",
    "worker_died": "Worker died",
    "worker_died_long": "The pipeline worker crashed:\n\n{message}",
    "diskspace_full": "Out of disk space; free some space and try again.",
    "permission_denied": "Permission denied.",
    "file_not_found": "File not found: {path}",
    "network_error": "Network error: {reason}",
    "app_about_title": "About Anime Upscaler GUI",
    "app_about_body":
        "Anime Upscaler GUI\n\n"
        "Tkinter desktop front-end for the upscale_anime toolkit.\n"
        "See docs/onboarding.md for usage and docs/themes.md for theming.",
    "menu_file": "File",
    "menu_edit": "Edit",
    "menu_run": "Run",
    "menu_tools": "Tools",
    "menu_help": "Help",
    "settings_saved": "Settings saved.",
    "settings_imported": "Settings imported from {name}",
    "settings_exported": "Settings exported to {name}",
    "queue_added": "Added {n} file(s).",
    "queue_loaded": "Loaded: {name}",
    "reset_confirm_title": "Reset",
    "reset_confirm_message": "Reset all settings to defaults?",
    "import_confirm_title": "Import settings",
    "import_confirm_message":
        "Replace current settings with the contents of:\n{path}\n\n"
        "Unknown fields will be ignored; missing fields keep their current values.",
    "move_confirm_title": "Move app data",
    "move_confirm_message":
        "Move all app data to:\n{path}\n\n"
        "Files at the old location ({old}) will be removed.",
    "move_done": "App data moved to: {path}",
    "move_failed": "Move failed",
    "export_dialog_title": "Export settings to...",
    "import_dialog_title": "Import settings from...",
    "move_dialog_title": "Move app data to...",
    "open_failed": "Cannot open",
    "models_tab": "Models",
    "upscale_tab": "Upscale",
}


_AR: Dict[str, str] = {
    "tainted_checkpoint_title": "نموذج ملوث",
    "tainted_checkpoint_message":
        "يبدو أن هذا النموذج ملوث (تحيّز الطباعة في النطاق [0.35, 0.45]).\n"
        "قد تكون المخرجات مسطحة. هل تريد المتابعة؟",
    "unsupported_kind": "{filename} نوعه {kind!r}؛ غير مدعوم في هذه النسخة.",
    "unsupported_preset": "{pid} نوعه {kind!r}؛ غير مفعّل في هذه النسخة.",
    "no_url": "الصق رابطاً أولاً.",
    "download_failed": "فشل التحميل",
    "import_failed": "فشل الاستيراد",
    "worker_died": "تعطّل العامل",
    "worker_died_long": "تعطّل عامل المعالجة:\n\n{message}",
    "diskspace_full": "لا توجد مساحة كافية على القرص؛ حرر مساحة وأعد المحاولة.",
    "permission_denied": "تم رفض الصلاحية.",
    "file_not_found": "الملف غير موجود: {path}",
    "network_error": "خطأ في الشبكة: {reason}",
    "app_about_title": "حول Anime Upscaler GUI",
    "app_about_body":
        "Anime Upscaler GUI\n\n"
        "واجهة سطح مكتب Tkinter لأدوات upscale_anime.\n"
        "راجع docs/onboarding.md للاستخدام و docs/themes.md للسمات.",
    "menu_file": "ملف",
    "menu_edit": "تحرير",
    "menu_run": "تشغيل",
    "menu_tools": "أدوات",
    "menu_help": "مساعدة",
    "settings_saved": "تم حفظ الإعدادات.",
    "settings_imported": "تم استيراد الإعدادات من {name}",
    "settings_exported": "تم تصدير الإعدادات إلى {name}",
    "queue_added": "تمت إضافة {n} ملف/ملفات.",
    "queue_loaded": "تم التحميل: {name}",
    "reset_confirm_title": "إعادة تعيين",
    "reset_confirm_message": "إعادة ضبط جميع الإعدادات إلى الافتراضية؟",
    "import_confirm_title": "استيراد الإعدادات",
    "import_confirm_message":
        "استبدال الإعدادات الحالية بمحتويات:\n{path}\n\n"
        "سيتم تجاهل الحقول غير المعروفة؛ تبقى القيم الافتراضية للحقول المفقودة.",
    "move_confirm_title": "نقل بيانات التطبيق",
    "move_confirm_message":
        "نقل جميع بيانات التطبيق إلى:\n{path}\n\n"
        "ستُحذف الملفات في الموقع القديم ({old}).",
    "move_done": "نُقلت بيانات التطبيق إلى: {path}",
    "move_failed": "فشل النقل",
    "export_dialog_title": "تصدير الإعدادات إلى...",
    "import_dialog_title": "استيراد الإعدادات من...",
    "move_dialog_title": "نقل بيانات التطبيق إلى...",
    "open_failed": "تعذّر الفتح",
    "models_tab": "النماذج",
    "upscale_tab": "تحسين",
}


_CATALOGS: Dict[str, Dict[str, str]] = {"en": _EN, "ar": _AR}


_active_locale: str = DEFAULT_LOCALE


def set_locale(name: str) -> None:
    """Switch the active locale. Unknown names are ignored (keep prior active).
    Ponytail: no future for catalog objects; just mutate one module attr."""
    global _active_locale
    if name in _CATALOGS:
        _active_locale = name


def active_locale() -> str:
    return _active_locale


def is_rtl() -> bool:
    """Arabic is the only RTL locale shipped today. Add to this set when adding languages."""
    return _active_locale == "ar"


def side_for(default: str = "left") -> str:
    """Return the mirrored side for RTL locales, else the default.
    Ponytail: caller passes the LTR side; no need to introspect widgets.
    """
    if is_rtl() and default in ("left", "right"):
        return "right" if default == "left" else "left"
    return default


def _format(text: str, kwargs: dict) -> str:
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError):
        return text


def _(key: str, **kwargs) -> str:
    """Return the localized string for `key`. Falls back to English, then to the key itself."""
    catalog = _CATALOGS.get(_active_locale, _EN)
    text = catalog.get(key) or _EN.get(key, key)
    return _format(text, kwargs)


def locale_choices() -> list:
    """Available locales for radio/radio UI."""
    return list(SUPPORTED_LOCALES)


# Test helper.
def _reset_for_tests() -> Callable[[], None]:
    """Restore DEFAULT_LOCALE. Returns an undo function."""
    global _active_locale
    prev = _active_locale
    _active_locale = DEFAULT_LOCALE

    def _undo() -> None:
        global _active_locale
        _active_locale = prev
    return _undo
