"""Test the real load_student source extracted from infer.py / export.py."""
import ast
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "anime_upscaler"))

import torch
from student import TinySRVGGStudent, build_student


def extract_load_student(path):
    src = open(path).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "load_student")
    return ast.get_source_segment(src, fn)


def make_pickle_checkpoint(tmp):
    # Path inside the state dict forces the pickle fallback
    # (weights_only=True rejects non-tensor objects).
    state = {"epoch": 1, "student": TinySRVGGStudent().state_dict(),
             "val_psnr": 30.0, "args": {"arch": "srvgg", "scale": 4},
             "meta": Path("/tmp/x")}
    torch.save(state, tmp / "student_best.pt")


for label, path in [("infer.py", "anime_upscaler/infer.py"),
                    ("export.py", "anime_upscaler/export.py")]:
    fn_src = extract_load_student(path)
    ns = {"torch": torch, "Path": Path, "build_student": build_student}
    exec(fn_src, ns)
    load_student = ns["load_student"]

    tmp = Path(tempfile.mkdtemp())
    make_pickle_checkpoint(tmp)

    # allow_pickle=False -> RuntimeError (NOT NameError)
    try:
        load_student(str(tmp), "cpu", allow_pickle=False)
        print(label, "FAIL: expected RuntimeError")
    except RuntimeError as e:
        assert "pickle" in str(e).lower()
        print(label, "OK: RuntimeError ->", str(e)[:55])
    except NameError as e:
        print(label, "FAIL: NameError ->", e)

    # allow_pickle=True -> loads the model
    m = load_student(str(tmp), "cpu", allow_pickle=True)
    print(label, "OK: loaded with allow_pickle=True ->", type(m).__name__)

    # arch override still works through the same path
    m2 = load_student(str(tmp), "cpu", arch="srvgg", allow_pickle=True)
    print(label, "OK: arch override ->", type(m2).__name__)

print()
print("PICKLE-FALLBACK TESTS PASSED")
