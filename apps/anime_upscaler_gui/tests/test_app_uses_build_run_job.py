"""Tests for the app.py -> build_run_job wiring (Phase B4).

Pins the contract that _enqueue_next (and any other future job-building
call site in app.py) goes through build_run_job instead of constructing a
RunJob literal directly. This is the B4 payoff: adding a new RunJob field
should require touching build_run_job + RunJob, not also _enqueue_next.
"""
import inspect

from apps.anime_upscaler_gui.anime_upscaler_gui import app


def test_app_imports_build_run_job():
    """app.py must import build_run_job and make_panel_value_provider."""
    assert hasattr(app, "_build_run_job")
    assert hasattr(app, "_make_panel_value_provider")
    assert callable(app._build_run_job)
    assert callable(app._make_panel_value_provider)


def test_app_does_not_construct_runjob_directly():
    """app.py source must NOT contain a RunJob(...) call.

    This is the structural invariant that makes Phase B4's refactor hold:
    the only place a RunJob is built is build_run_job. Adding a new
    RunJob field becomes a one-place change.
    """
    src = inspect.getsource(app)
    # Allow the name "RunJob" in comments/docstrings (e.g. "# RunJob.is_video"),
    # but not the call form `RunJob(`:
    assert "RunJob(" not in src, (
        "app.py still constructs RunJob(...) directly; "
        "route it through build_run_job instead.",
    )


def test_enqueue_next_calls_build_run_job():
    """_enqueue_next body must call _build_run_job, not RunJob(...)."""
    src = inspect.getsource(app.UpscaleGUI._enqueue_next)
    assert "_build_run_job(" in src, "_enqueue_next did not call _build_run_job"
    # The old form was a 30-arg RunJob( ... ) literal with many `name=` keyword
    # arguments. After Phase B4 those names should only appear as kwargs to
    # _build_run_job(...). Strip comments + strings, then assert NO bare
    # `name=` kwargs remain (those would indicate a literal RunJob construction).
    import re
    code_only = re.sub(r"#[^\n]*\n", "\n", src)  # strip line comments
    code_only = re.sub(r'"""[^"]*"""', "", code_only, flags=re.DOTALL)  # strip docstrings
    # A literal RunJob( job_id=... ) call would have these as bare kwargs at
    # call site. After B4 they should only appear inside _build_run_job(...) args.
    forbidden = [
        "RunJob(",
        "job_id=",
        "input_path=",
        "output_path=",
        "model_filename=",
        "outscale=",
        "fp16=",
        "decode=",
        "downscale_max_edge=",
        "tile_size=",
        "tta=",
        "use_tensorrt=",
        "nvenc_qp=",
        "cascade_mode=",
        "cut_start_seconds=",
    ]
    for kw in forbidden:
        assert kw not in code_only, (
            f"_enqueue_next still has bare '{kw}' outside a comment; "
            "build_run_job should be the only RunJob builder.",
        )


def test_enqueue_next_uses_panel_value_provider():
    """_enqueue_next must call make_panel_value_provider once at the top."""
    src = inspect.getsource(app.UpscaleGUI._enqueue_next)
    assert "_make_panel_value_provider(" in src, (
        "_enqueue_next should wire the live panel via make_panel_value_provider"
    )


def test_app_no_longer_imports_runjob():
    """RunJob should not be a top-level app.py import anymore."""
    src = inspect.getsource(app)
    # The first 100 lines should have the imports. RunJob must be absent:
    head = "\n".join(src.splitlines()[:120])
    assert "import RunJob" not in head and " RunJob," not in head and ", RunJob " not in head, (
        "app.py still imports RunJob; after Phase B4 it should not."
    )


def test_enqueue_next_resolves_is_video_via_path_fallback():
    """_enqueue_next must still fall back to _is_video_path when j.is_video is False."""
    src = inspect.getsource(app.UpscaleGUI._enqueue_next)
    assert "_is_video_path" in src, (
        "_enqueue_next lost the _is_video_path fallback; video files without",
        "j.is_video set will not get batch_size=1.",
    )


def test_enqueue_next_signature_unchanged():
    """_enqueue_next signature must still take a single jobs_to_run arg."""
    sig = inspect.signature(app.UpscaleGUI._enqueue_next)
    params = list(sig.parameters.keys())
    assert params == ["self", "jobs_to_run"], f"got {params}"
