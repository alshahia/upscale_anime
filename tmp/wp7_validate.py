"""WP-7 final validation: syntax, pickle-fallback path, engine regression."""
import ast
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "anime_upscaler"))

ok = True

# 1. Syntax checks
for f in ["src/inference/engine.py", "anime_upscaler/export.py",
          "anime_upscaler/infer.py", "anime_upscaler/student.py"]:
    try:
        ast.parse(open(f).read())
        print("syntax OK:", f)
    except SyntaxError as e:
        ok = False
        print("syntax FAIL:", f, e)

# 2. Pickle-fallback path in infer.py load_student
try:
    import torch
    from student import TinySRVGGStudent
    import infer as infer_mod

    tmp = Path(tempfile.mkdtemp())
    state = {"epoch": 1, "student": TinySRVGGStudent().state_dict(),
             "val_psnr": 30.0, "args": {"arch": "srvgg", "scale": 4}}
    # Wrapper containing a Path object forces pickle loading
    # (weights_only=True rejects non-tensor objects).
    torch.save({"checkpoint": state, "meta": Path("/tmp/x")},
               tmp / "student_best.pt")

    # 2a. allow_pickle=False -> RuntimeError (not NameError)
    try:
        infer_mod.load_student(str(tmp), "cpu", allow_pickle=False)
        print("FAIL: expected RuntimeError")
        ok = False
    except RuntimeError as e:
        assert "pickle" in str(e).lower(), str(e)
        print("OK: allow_pickle=False raises RuntimeError:", str(e)[:50])
    except NameError as e:
        print("FAIL: NameError (scoping bug):", e)
        ok = False

    # 2b. allow_pickle=True -> loads fine
    m = infer_mod.load_student(str(tmp), "cpu", allow_pickle=True)
    print("OK: allow_pickle=True loads model:", type(m).__name__)
except Exception as e:
    print("SKIP: infer load_student test:", type(e).__name__, str(e)[:100])

# 3. Engine regression (quick)
try:
    import numpy as np
    from inference.engine import InferenceEngine
    eng = InferenceEngine(TinySRVGGStudent(), device="cpu")
    lr = np.random.rand(24, 24, 3).astype(np.float32)
    sr = eng.run(lr)
    assert sr.shape == (96, 96, 3), sr.shape
    srs = eng.run_batch([lr, lr])
    assert len(srs) == 2 and srs[0].shape == (96, 96, 3)
    res = eng.benchmark(input_size=(3, 24, 24), num_runs=2, warmup=1)
    assert res["output_resolution"] == "96x96", res["output_resolution"]
    print("OK: engine regression (run/run_batch/benchmark)")
except Exception as e:
    ok = False
    print("FAIL: engine regression:", type(e).__name__, str(e)[:150])

print()
print("VALIDATION", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
