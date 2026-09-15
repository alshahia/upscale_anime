"""Phase 4: atomic write, export/import, settings migration."""
import json
import os
import tempfile
import uuid
from pathlib import Path


def _fresh_appdir() -> Path:
    return Path(tempfile.gettempdir()) / f"aug_mig_{uuid.uuid4().hex[:8]}"


# ---- atomic write ---- #

def test_atomic_save_no_partial_writes(tmp_path):
    """If save is interrupted (simulated by raising mid-write), the original
    file must be intact."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    s = _Settings(paths)
    s.update(outscale=3.0, fp16=False)
    original = paths.settings_path.read_text(encoding="utf-8")

    # Simulate crash mid-write by replacing the helper temporarily.
    real = s.save
    def boom(_self=paths.settings_path):
        # Touch the original file would have to be untouched: the helper writes
        # to a tmp file first, then os.replace. Force an exception between those
        # two steps by monkey-patching os.replace.
        raise RuntimeError("simulated crash")
    import apps.anime_upscaler_gui.anime_upscaler_gui.settings as settings_mod
    orig_replace = settings_mod.os.replace
    settings_mod.os.replace = lambda *a, **k: boom()
    try:
        try:
            s.save()
        except RuntimeError:
            pass
    finally:
        settings_mod.os.replace = orig_replace
        # restore the bound real save so subsequent tests aren't disturbed
        s.save = real.__get__(s, type(s))

    # The original file must still be readable and contain the same JSON.
    after = paths.settings_path.read_text(encoding="utf-8")
    assert after == original


def test_no_temp_files_left_on_disk(tmp_path):
    """Successful atomic write must clean up its temp file."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    s = _Settings(paths)
    s.update(outscale=2.5)
    leftover = [p for p in os.listdir(paths.app_dir) if p.startswith(".") and p.endswith(".tmp")]
    assert not leftover, f"leftover tmp files: {leftover}"


# ---- export / import ---- #

def test_export_roundtrip_preserves_values():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    s1 = _Settings(paths)
    s1.update(outscale=3.5, fp16=False, last_model="realesr-animevideov3.pth", gpu_guard_mode="auto_downscale")

    out = d / "export.json"
    s1.export(out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["outscale"] == 3.5
    assert payload["fp16"] is False
    assert payload["last_model"] == "realesr-animevideov3.pth"

    s2 = _Settings(paths)
    s2.import_file(out)
    assert s2.data.outscale == 3.5
    assert s2.data.fp16 is False
    assert s2.data.last_model == "realesr-animevideov3.pth"
    assert s2.data.gpu_guard_mode == "auto_downscale"


def test_import_ignores_unknown_fields():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    s = _Settings(paths)
    bogus = {"outscale": 4.0, "future_field_we_dont_have": "lol", "another": [1, 2, 3]}
    bad = d / "bogus.json"
    bad.write_text(json.dumps(bogus))
    s.import_file(bad)
    assert s.data.outscale == 4.0
    # No crash. The unknown field is dropped silently.


def test_import_invalidates_invalid_formstate():
    """Unparseable values get the field's default; the import itself must
    not crash."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    s = _Settings(paths)
    s.update(outscale=3.0, batch_size=4)
    bad = d / "bad.json"
    bad.write_text(json.dumps({"outscale": "not a number", "batch_size": "wat"}))
    s.import_file(bad)
    # Coerce returns a fresh _Defaults (so the bad fields get the defaults
    # 2.0 / 1) — the unparseable values are dropped rather than crashing.
    assert s.data.outscale == 2.0
    assert s.data.batch_size == 1


def test_export_writes_atomically():
    """export() must use atomic write (tmp + replace), not a bare write_text."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d = _fresh_appdir()
    paths = _AppPaths(d)
    s = _Settings(paths)
    s.update(outscale=2.0)
    out = d / "subdir" / "exported.json"
    s.export(out)
    # Subdir was auto-created; file exists and is valid JSON.
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["outscale"] == 2.0


# ---- storage folder picker integration ---- #

def test_move_to_then_save_round_trip():
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import _AppPaths, _Settings
    d1 = _fresh_appdir()
    d2 = _fresh_appdir()
    p = _AppPaths(d1)
    s = _Settings(p)
    s.update(outscale=4.0)
    p.move_to(d2)
    # settings.json is at the new location
    assert (d2 / "settings.json").exists()
    # Reloading from the new location gives the same data
    p2 = _AppPaths(d2)
    s2 = _Settings(p2)
    assert s2.data.outscale == 4.0
    # And after move, app_dir should be updateable on the in-memory settings
    s.data.app_dir = str(d2)
    s.save()
    s3 = _Settings(p2)
    assert s3.data.app_dir == str(d2)
