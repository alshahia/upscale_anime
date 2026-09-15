"""Phase 7: error mapping (URLError / OSError / generic)."""
import errno

from apps.anime_upscaler_gui.anime_upscaler_gui.errors import humanize


def test_humanize_url_error():
    from urllib.error import URLError
    e = URLError("no internet")
    msg = humanize(e)
    assert "Network error" in msg
    assert "no internet" in msg


def test_humanize_disk_full_posix():
    e = OSError(errno.ENOSPC, "No space left on device")
    msg = humanize(e)
    assert "disk space" in msg.lower()


def test_humanize_disk_full_windows():
    e = OSError(112, "There is not enough space on the disk.")
    msg = humanize(e)
    assert "disk space" in msg.lower()


def test_humanize_permission_denied():
    e = OSError(errno.EACCES, "Permission denied", "/etc/shadow")
    msg = humanize(e)
    assert "ermission" in msg


def test_humanize_file_not_found():
    e = OSError(errno.ENOENT, "missing", "/tmp/x")
    msg = humanize(e)
    assert "not found" in msg.lower()


def test_humanize_runtime_error_passthrough():
    e = RuntimeError("CUDA out of memory")
    msg = humanize(e)
    assert "RuntimeError" in msg
    assert "CUDA" in msg


def test_humanize_value_error_passthrough():
    e = ValueError("bad input")
    msg = humanize(e)
    assert "ValueError" in msg
    assert "bad input" in msg


def test_humanize_unknown_type_uses_class_name():
    class _CustomError(Exception):
        pass
    e = _CustomError("weird")
    msg = humanize(e)
    assert "_CustomError" in msg
    assert "weird" in msg
