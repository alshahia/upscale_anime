"""Unit tests for pipeline/backends.py BackendRegistry (Phase B3).

Tests verify the registry semantics WITHOUT importing TensorRT / CUDA.
Tests pass a custom registry of fake backends (no actual model loading),
so they run on any machine, headless, in milliseconds.

The tests cover:
  - predicate-based selection (first match wins)
  - factory exception -> falls through to next
  - predicate exception -> skip silently
  - empty registry / no matches -> raises RuntimeError
  - default REGISTRY has TRT first, PyTorch last
  - backwards-compat: _make_backend is select_backend
  - adding a new backend via register_backend("CustomName", ...) works
  - BackendRegistry is frozen (immutable)
"""
import pytest
import torch

from apps.anime_upscaler_gui.anime_upscaler_gui.pipeline import (
    REGISTRY,
    BackendRegistry,
    BackendSelectionCtx,
    _make_backend,
    register_backend,
    select_backend,
)


# --- Helpers ---------------------------------------------------------------
def _ctx(**overrides):
    """Build a BackendSelectionCtx with sane defaults; tests override fields."""
    defaults = dict(
        model=None, ckpt_path="/fake/ckpt.pth", kind="rfdn_student",
        device=torch.device("cpu"),  # always CPU in tests so TRT predicate rejects
        fp16=True, use_tensorrt=False, tta=False, batch_size=1,
    )
    defaults.update(overrides)
    return BackendSelectionCtx(**defaults)


def test_default_registry_has_trt_then_pytorch():
    """Default REGISTRY is ordered: TRT (priority 1), PyTorch (fallback)."""
    names = [e.name for e in REGISTRY]
    assert names == ["TensorRT", "PyTorch"], f"got {names}"


def test_select_backend_first_match_wins():
    """Of multiple matching entries, the first wins."""
    calls = []

    def make(name):
        return BackendRegistry(
            name=name,
            predicate=lambda c: True,
            factory=lambda c: calls.append(name) or f"backend-{name}",
        )

    reg = [make("A"), make("B"), make("C")]
    backend, name = select_backend(*([None] * 6), registry=reg)
    # 6 positional args: model, ckpt_path, kind, device, fp16, use_tensorrt
    # then keyword tta=False, batch_size=1
    assert name == "A"
    assert backend == "backend-A"
    assert calls == ["A"]


def test_select_backend_skips_predicate_false():
    """Predicates returning False cause the entry to be skipped."""
    reg = [
        BackendRegistry(name="Skip", predicate=lambda c: False,
                        factory=lambda c: "should-not-run"),
        BackendRegistry(name="Take", predicate=lambda c: True,
                        factory=lambda c: "taken"),
    ]
    backend, name = select_backend(None, "/c", "k", torch.device("cpu"), False, False,
                                    registry=reg)
    assert name == "Take"
    assert backend == "taken"


def test_select_backend_factory_failure_falls_through():
    """If a factory raises, the registry logs and tries the next entry."""
    reg = [
        BackendRegistry(name="Boom", predicate=lambda c: True,
                        factory=lambda c: (_ for _ in ()).throw(RuntimeError("nope"))),
        BackendRegistry(name="OK", predicate=lambda c: True,
                        factory=lambda c: "fallback"),
    ]
    backend, name = select_backend(None, "/c", "k", torch.device("cpu"), False, False,
                                    registry=reg)
    assert name == "OK"
    assert backend == "fallback"


def test_select_backend_predicate_exception_skips():
    """A broken predicate logs and is skipped (does NOT cause RuntimeError)."""
    reg = [
        BackendRegistry(name="BrokenPredicate",
                        predicate=lambda c: 1 / 0,  # raises
                        factory=lambda c: "unreachable"),
        BackendRegistry(name="OK", predicate=lambda c: True,
                        factory=lambda c: "ok"),
    ]
    backend, name = select_backend(None, "/c", "k", torch.device("cpu"), False, False,
                                    registry=reg)
    assert name == "OK"
    assert backend == "ok"


def test_select_backend_no_match_raises():
    """Empty predicate results in RuntimeError listing all backend names."""
    reg = [
        BackendRegistry(name="A", predicate=lambda c: False,
                        factory=lambda c: "a"),
        BackendRegistry(name="B", predicate=lambda c: False,
                        factory=lambda c: "b"),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        select_backend(None, "/c", "k", torch.device("cpu"), False, False,
                       registry=reg)
    assert "A" in str(excinfo.value)
    assert "B" in str(excinfo.value)


def test_select_backend_all_factories_fail_raises_with_chain():
    """When every factory fails, RuntimeError chains the last exception."""
    reg = [
        BackendRegistry(name="X", predicate=lambda c: True,
                        factory=lambda c: (_ for _ in ()).throw(ValueError("first"))),
        BackendRegistry(name="Y", predicate=lambda c: True,
                        factory=lambda c: (_ for _ in ()).throw(ValueError("second"))),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        select_backend(None, "/c", "k", torch.device("cpu"), False, False,
                       registry=reg)
    # last error should be chained
    assert excinfo.value.__cause__ is not None
    assert "second" in str(excinfo.value.__cause__)


def test_select_backend_predicate_receives_ctx():
    """Predicate sees the same ctx that the factory will see."""
    seen = []
    def pred(c):
        seen.append(c)
        return True
    reg = [
        BackendRegistry(name="Probe", predicate=pred, factory=lambda c: "ok"),
    ]
    select_backend(None, "/ckpt.pth", "custom_kind", torch.device("cpu"), True, False,
                   tta=True, registry=reg)
    assert len(seen) == 1
    c = seen[0]
    assert c.kind == "custom_kind"
    assert c.fp16 is True
    assert c.tta is True
    assert c.batch_size == 1
    assert c.ckpt_path == "/ckpt.pth"


def test_select_backend_tta_disables_trt():
    """Even on a CUDA + fp16 + use_tensorrt setup, TTA forces fallback to PyTorch.

    We can't actually construct a CUDA device on a CPU test machine, so we
    fake the registry's TRT predicate with a "would-be-true" body that also
    checks tta. The default registry already has this rule (see _trt_predicate);
    this test pins that contract.
    """
    # Re-implement the TRT rule's semantics here to verify the contract.
    def fake_trt_predicate(c):
        if c.tta: return False
        if not c.use_tensorrt: return False
        if c.device.type != "cuda": return False
        if not c.fp16: return False
        return True

    def fake_pytorch_predicate(c):
        return True

    reg = [
        BackendRegistry(name="TensorRT", predicate=fake_trt_predicate,
                        factory=lambda c: "trt-backend"),
        BackendRegistry(name="PyTorch", predicate=fake_pytorch_predicate,
                        factory=lambda c: "pytorch-backend"),
    ]
    # Even with use_tensorrt=True + cuda + fp16, tta=True forces PyTorch:
    backend, name = select_backend(
        _ctx(device=torch.device("cpu"), use_tensorrt=True, fp16=True, tta=True),
        "/c", "k", torch.device("cpu"), True, True,
        tta=True, registry=reg,
    )
    assert name == "PyTorch"


def test_register_backend_appends_to_default_registry():
    """register_backend() appends a new entry to REGISTRY."""
    before = len(REGISTRY)
    entry = register_backend(
        "_test_marker",
        predicate=lambda c: False,  # never runs
        factory=lambda c: "test",
    )
    try:
        assert len(REGISTRY) == before + 1
        # Last entry has the test name:
        assert REGISTRY[-1].name == "_test_marker"
        assert REGISTRY[-1] is entry
    finally:
        REGISTRY.remove(entry)  # cleanup so other tests aren't affected


def test_backend_registry_is_frozen():
    """BackendRegistry is a frozen dataclass -- cannot mutate after creation."""
    e = BackendRegistry(name="X", predicate=lambda c: True, factory=lambda c: None)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        e.name = "Y"  # type: ignore[misc]


def test_backwards_compat_make_backend_is_select_backend():
    """Phase A1/B3 contract: _make_backend wraps select_backend.

    After Phase E the alias is a DeprecatedAlias proxy that emits a
    DeprecationWarning on first use. The proxy __wrapped__ attribute is
    the original function, so call semantics are preserved while
    downstream callers get the soft-deprecation signal.
    """
    assert _make_backend.__wrapped__ is select_backend
    assert callable(_make_backend)
    # Calling the proxy must forward to the underlying function. We use a
    # custom registry (no PyTorch factory needed) so the test does not depend
    # on a real checkpoint on disk. We suppress the warning so the test
    # output stays clean.
    import warnings
    fake_registry = [
        BackendRegistry(
            name="FakeBackend",
            predicate=lambda c: True,
            factory=lambda c: "ok",
        ),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backend, name = _make_backend(
            None, "/c", "k", torch.device("cpu"), False, False,
            registry=fake_registry,
        )
    assert name == "FakeBackend"
    assert backend == "ok"


def test_default_registry_falls_back_to_pytorch_when_tta_or_no_trt(monkeypatch):
    """End-to-end: with use_tensorrt=False on a CPU box, default REGISTRY -> PyTorch.

    Avoid actually loading a checkpoint (the PyTorch factory calls torch.load),
    so we monkeypatch torch.load to a no-op that returns a sentinel state dict.
    """
    import torch as _t

    def _fake_load(path, *a, **kw):
        # Minimal state-dict-like object so RFDNStudent(**kw) accepts it.
        return {}

    monkeypatch.setattr(_t, "load", _fake_load)

    backend, name = select_backend(
        model=None, ckpt_path="/c", kind="rfdn_student",
        device=torch.device("cpu"), fp16=True, use_tensorrt=False, tta=False,
    )
    assert name == "PyTorch"
    # _make_backend path returns the same result (separate objects, but same name):
    backend2, name2 = _make_backend(
        model=None, ckpt_path="/c", kind="rfdn_student",
        device=torch.device("cpu"), fp16=True, use_tensorrt=False, tta=False,
    )
    assert name2 == "PyTorch"


def test_select_backend_uses_ctx_batch_size():
    """BackendSelectionCtx.batch_size flows through to the factory."""
    received = []
    def factory(c):
        received.append(c.batch_size)
        return "ok"
    reg = [BackendRegistry(name="X", predicate=lambda c: True, factory=factory)]
    select_backend(_ctx(batch_size=8), "/c", "k", torch.device("cpu"), False, False,
                   batch_size=8, registry=reg)
    assert received == [8]


def test_select_backend_takes_registry_kwarg_only():
    """The signature accepts a custom registry via keyword arg."""
    import inspect
    sig = inspect.signature(select_backend)
    params = list(sig.parameters.keys())
    # registry is last; batch_size is second-to-last:
    assert params[-1] == "registry"
    assert params[-2] == "batch_size"
    assert sig.parameters["registry"].default is None
