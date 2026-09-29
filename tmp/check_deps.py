import importlib
for m in ["pandas", "openpyxl", "pyiqa", "onnxruntime", "imageio", "av"]:
    try:
        __import__(m)
        print(m, "OK")
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
