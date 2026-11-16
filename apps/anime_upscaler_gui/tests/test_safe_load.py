"""Checkpoint-loading safety tests (Task 14).

Threat model: a .pth file is a pickle. Loading one with weights_only=False
can execute arbitrary code, so the GUI must
  (a) never execute pickled payloads while scanning pretrained/ for the
      installed-model list, and
  (b) only fall back to pickle loading of one explicit checkpoint after the
      user has deliberately opted in with
      ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE=1 -- and then warn loudly.

The opt-in regression check uses a *harmless* legacy payload (a plain
module-level marker class with no side effects): a plain tensor pickle would
already load through the weights_only=True path and would not exercise the
fallback at all.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from apps.anime_upscaler_gui.anime_upscaler_gui import arch_registry as archs
from apps.anime_upscaler_gui.anime_upscaler_gui import registry as registry_mod
from apps.anime_upscaler_gui.anime_upscaler_gui.registry import (
    InstalledModel,
    ModelRegistry,
    _load_preset_catalog,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.services.model_service import (
    format_dropdown_label,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_PRETRAINED = REPO_ROOT / "pretrained"
UNSAFE_ENV = "ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE"


class _HarmlessLegacyMeta:
    """Picklable, side-effect-free object that weights_only=True refuses."""

    def __init__(self, tag: str = "legacy-checkpoint") -> None:
        self.tag = tag


def _write_malicious_pickle(path: Path, marker: Path) -> None:
    """Write a checkpoint whose unpickling writes marker (the payload)."""

    class Evil:
        def __reduce__(self):
            return (Path.write_text, (marker, "pwned"))

    torch.save(Evil(), path)


def _write_weights_only_safe_stub(path: Path) -> None:
    """A trivially safe checkpoint (tensor only, no wrapper metadata)."""
    torch.save({"body.0.weight": torch.zeros(1)}, path)


def _write_harmless_legacy_checkpoint(path: Path) -> None:
    """A realistic legacy checkpoint: state dict + pickled metadata object."""
    from apps.anime_upscaler_gui.anime_upscaler_gui._models.students import RFDN

    torch.save(
        {"params": RFDN().state_dict(), "legacy_meta": _HarmlessLegacyMeta()},
        path,
    )


@pytest.fixture(autouse=True)
def _safe_default_env(monkeypatch):
    """Every test starts from the safe default: no unsafe opt-in."""
    monkeypatch.delenv(UNSAFE_ENV, raising=False)


# --------------------------------------------------------------------------- #
# Registry.scan_installed: never executes pickled payloads
# --------------------------------------------------------------------------- #
def test_scan_does_not_execute_a_pickled_payload(tmp_path):
    marker = tmp_path / "pwned.txt"
    evil = tmp_path / "evil.pth"
    _write_malicious_pickle(evil, marker)

    reg = ModelRegistry(tmp_path)
    installed = reg.scan_installed()

    assert not marker.exists(), (
        "scan_installed executed pickled code from a checkpoint: a plain "
        "directory refresh must never run arbitrary code"
    )
    assert [m.filename for m in installed] == ["evil.pth"]
    assert installed[0].kind is None
    assert installed[0].supported is False


def test_scan_records_skip_reason_for_unsafe_checkpoint(tmp_path):
    marker = tmp_path / "pwned.txt"
    evil = tmp_path / "evil.pth"
    _write_malicious_pickle(evil, marker)

    reg = ModelRegistry(tmp_path)
    reg.scan_installed()

    assert not marker.exists()
    assert len(reg._skipped) == 1, "unsafe checkpoints must be surfaced, not hidden"
    path, reason = reg._skipped[0]
    assert path.endswith("evil.pth")
    assert reason, "skip record must carry a diagnostic reason"


def test_scan_skips_legacy_pickle_checkpoint_without_executing(tmp_path):
    ckpt = tmp_path / "legacy.pth"
    _write_harmless_legacy_checkpoint(ckpt)

    reg = ModelRegistry(tmp_path)
    models = reg.scan_installed()

    assert [m.filename for m in models] == ["legacy.pth"]
    assert models[0].kind is None, "scan must not classify via an unsafe load"
    assert reg._skipped and reg._skipped[0][0].endswith("legacy.pth")


def test_scan_never_honours_the_optin_env_var(tmp_path, monkeypatch):
    # Opt-in is for one deliberate load; a bulk directory scan stays safe.
    monkeypatch.setenv(UNSAFE_ENV, "1")
    marker = tmp_path / "pwned.txt"
    evil = tmp_path / "evil.pth"
    _write_malicious_pickle(evil, marker)

    ModelRegistry(tmp_path).scan_installed()

    assert not marker.exists()


# --------------------------------------------------------------------------- #
# arch_registry: default-safe load, explicit env-var opt-in
# --------------------------------------------------------------------------- #
def test_arch_loader_refuses_malicious_pickle_by_default(tmp_path):
    marker = tmp_path / "pwned.txt"
    evil = tmp_path / "evil.pth"
    _write_malicious_pickle(evil, marker)

    with pytest.raises(RuntimeError, match=UNSAFE_ENV):
        archs.build("rfdn_student", evil)

    assert not marker.exists(), "refused checkpoint must not have executed code"


def test_arch_loader_refuses_legacy_pickle_by_default(tmp_path):
    ckpt = tmp_path / "legacy.pth"
    _write_harmless_legacy_checkpoint(ckpt)

    with pytest.raises(RuntimeError, match=UNSAFE_ENV):
        archs.build("rfdn_student", ckpt)


def test_arch_loader_opt_in_allows_harmless_legacy_checkpoint(tmp_path, monkeypatch):
    ckpt = tmp_path / "legacy.pth"
    _write_harmless_legacy_checkpoint(ckpt)
    monkeypatch.setenv(UNSAFE_ENV, "1")

    with pytest.warns(UserWarning) as record:
        model = archs.build("rfdn_student", ckpt)

    assert isinstance(model, nn.Module)
    assert model.upscale == 4
    messages = [str(w.message) for w in record]
    assert any("UNSAFE" in m for m in messages), "opt-in must warn loudly"
    assert any(UNSAFE_ENV in m for m in messages), (
        "warning must name the opt-in env var so the user knows how it was allowed"
    )


def test_arch_loader_opt_in_env_default_false_does_not_load_pickle(tmp_path, monkeypatch):
    ckpt = tmp_path / "legacy.pth"
    _write_harmless_legacy_checkpoint(ckpt)

    monkeypatch.setenv(UNSAFE_ENV, "true")  # only the exact value "1" enables it

    with pytest.raises(RuntimeError, match=UNSAFE_ENV):
        archs.build("rfdn_student", ckpt)


def test_load_state_dict_allow_unsafe_pickle_kwarg(tmp_path):
    ckpt = tmp_path / "legacy.pth"
    _write_harmless_legacy_checkpoint(ckpt)

    with pytest.warns(UserWarning, match="UNSAFE"):
        state = archs._load_state_dict(ckpt, allow_unsafe_pickle=True)

    assert isinstance(state, dict)
    assert "legacy_meta" not in state
    assert any(key.startswith(("head.", "blocks.")) for key in state)


# --------------------------------------------------------------------------- #
# Real shipped checkpoints still load safely and end to end
# --------------------------------------------------------------------------- #
_SHIPPED_KINDS = {
    "realesr-animevideov3.pth": "srvgg",
    "4xLSDIRCompactv2.pth": "srvgg",
    "span_pix_pretrain_4x.pth": "span",
    "span_mssim_pretrain_4x.pth": "span",
    "eranet_N12_pretrain_325k.pth": "era",
    "AnimeSR_v2.pth": "animesr",
    "RFDN_distill_v1_4x_student.pth": "rfdn_student",
    "SRVGG_distill_v1_4x_student.pth": "srvgg_student",
}


def test_real_shipped_checkpoints_load_with_weights_only(tmp_path):
    if not REAL_PRETRAINED.is_dir():
        pytest.skip("pretrained/ checkpoints not present in this checkout")
    expected = {name: kind for name, kind in _SHIPPED_KINDS.items()
                if (REAL_PRETRAINED / name).is_file()}
    if not expected:
        pytest.skip("no shipped checkpoints present")

    reg = ModelRegistry(REAL_PRETRAINED)
    installed = {m.filename: m for m in reg.scan_installed()}
    skipped = {Path(path).name: err for path, err in reg._skipped}

    for name, kind in expected.items():
        assert name not in skipped, f"{name} unexpectedly failed safe load: {skipped[name]}"
        model = installed[name]
        assert model.kind == kind
        assert model.supported is True
        assert model.verified is True, f"{name} has a declared sha256 in the catalog"


@pytest.mark.parametrize("filename,kind", [
    ("RFDN_distill_v1_4x_student.pth", "rfdn_student"),
    ("realesr-animevideov3.pth", "srvgg"),
])
def test_real_checkpoint_builds_and_forwards(filename, kind):
    ckpt = REAL_PRETRAINED / filename
    if not ckpt.is_file():
        pytest.skip(f"pretrained/{filename} not present")

    model = archs.build(kind, ckpt)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 16, 16))
    assert tuple(out.shape) == (1, 3, 64, 64)


# --------------------------------------------------------------------------- #
# Preset catalog integrity: every preset either declares sha256 or is shown
# unverified in the UI
# --------------------------------------------------------------------------- #
def test_preset_catalog_requires_declared_sha256_or_documented_exception():
    entries = _load_preset_catalog()
    missing = sorted(e.id for e in entries if not e.verified)
    assert missing == ["RealESRGAN_x4plus_anime_6B"], (
        "only the documented Real-ESRGAN 6B preset may lack a sha256; add a "
        "hash to every other catalog entry"
    )


def test_catalog_warns_about_preset_without_sha256(caplog):
    with caplog.at_level(logging.WARNING, logger=registry_mod.__name__):
        _load_preset_catalog()
    assert "RealESRGAN_x4plus_anime_6B" in caplog.text


def test_preset_verified_property(tmp_path):
    reg = ModelRegistry(tmp_path)
    known = reg.preset_by_id("realesr-animevideov3")
    assert known.verified is True
    assert len(known.sha256) == 64
    unhashed = reg.preset_by_id("RealESRGAN_x4plus_anime_6B")
    assert unhashed.verified is False
    assert unhashed.sha256 == ""


def test_installed_models_carry_verified_flag_from_catalog(tmp_path):
    pre = tmp_path / "pretrained"
    pre.mkdir()
    _write_weights_only_safe_stub(pre / "realesr-animevideov3.pth")
    _write_weights_only_safe_stub(pre / "RealESRGAN_x4plus_anime_6B.pth")
    _write_weights_only_safe_stub(pre / "my_custom_upload.pth")

    reg = ModelRegistry(pre)
    by_name = {m.filename: m for m in reg.scan_installed()}

    assert by_name["realesr-animevideov3.pth"].verified is True
    assert by_name["RealESRGAN_x4plus_anime_6B.pth"].verified is False
    assert by_name["my_custom_upload.pth"].verified is False


def test_unverified_installed_model_is_marked_in_ui_labels(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.services.model_service import (
        TAG_UNVERIFIED,
    )
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.models_panel import (
        _installed_label,
    )

    unverified = InstalledModel(
        filename="mystery.pth", path=tmp_path / "mystery.pth",
        kind="srvgg", scale=4, size_mb=2.0, verified=False,
    )
    assert "[unverified]" in _installed_label(unverified)
    assert TAG_UNVERIFIED in format_dropdown_label(unverified)

    verified = InstalledModel(
        filename="realesr-animevideov3.pth",
        path=tmp_path / "realesr-animevideov3.pth",
        kind="srvgg", scale=4, size_mb=2.5, verified=True,
    )
    assert "[unverified]" not in _installed_label(verified)
    assert TAG_UNVERIFIED not in format_dropdown_label(verified)


def test_preset_without_hash_is_marked_in_preset_list(tmp_path):
    from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.models_panel import (
        _preset_label,
    )

    reg = ModelRegistry(tmp_path)
    unverified = _preset_label(reg.preset_by_id("RealESRGAN_x4plus_anime_6B"))
    assert "[unverified]" in unverified
    verified = _preset_label(reg.preset_by_id("realesr-animevideov3"))
    assert "[unverified]" not in verified
