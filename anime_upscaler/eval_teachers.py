# anime_upscaler/eval_teachers.py
"""Benchmark candidate TEACHER upscalers on the project test split.

Every candidate .pth is loaded through spandrel (architecture auto-detected,
extra arch pack installed) and evaluated on the exact seeded 99-image test
split / 384 px center crops that distill.py validation uses, with the same
psnr01/ssim01 implementations - so results are directly comparable with the
recorded baselines (bicubic 29.45 dB, incumbent SPAN teacher 30.44 dB).

Also reports parameter count, inference latency at train-crop size (48->192)
and eval size (96->384), and a Laplacian-variance sharpness score, so the
distillation teacher can be picked on measured evidence.

Usage:
    python anime_upscaler/eval_teachers.py --limit 16   # quick smoke
    python anime_upscaler/eval_teachers.py              # full test split
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AnimePairDataset, denorm01

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_DIR = ROOT / "checkpoints" / "candidates"
INCUMBENT_SPAN = ROOT / "checkpoints" / "NEOSR_SPAN_V7_ANIME_001" / "finetune_best.pth"


def psnr01(a, b):
    """Batch PSNR on [0,1] tensors, averaged over the batch (mirrors distill.py)."""
    mse = ((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean(dim=(1, 2, 3))
    mse = mse.clamp(min=1e-10)
    return (-10 * torch.log10(mse)).mean().item()


def ssim01(a, b):
    """Mean SSIM over batch using skimage (mirrors distill.py)."""
    from skimage.metrics import structural_similarity
    a = a.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    b = b.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    vals = [
        structural_similarity(ai, bi, channel_axis=2, data_range=1.0)
        for ai, bi in zip(a, b)
    ]
    return float(np.mean(vals))


_LAP_K = None


def lap_var01(x):
    """Mean squared Laplacian response of [0,1] NCHW tensor (sharpness proxy)."""
    global _LAP_K
    if _LAP_K is None or _LAP_K.device != x.device:
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=x.device)
        _LAP_K = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    resp = F.conv2d(x.clamp(0, 1), _LAP_K, padding=1, groups=3)
    return resp.pow(2).mean().item()


@torch.no_grad()
def predict(net, lr01, domain):
    """Run a spandrel-wrapped model honoring the probed input domain."""
    if domain == "-11":
        out = net(lr01 * 2.0 - 1.0)
        return ((out + 1.0) * 0.5).clamp(0, 1)
    return net(lr01).clamp(0, 1)


@torch.no_grad()
def probe_domain(net, loader, device, max_batches=4):
    """Pick input convention ([0,1] vs [-1,1]) by which scores higher PSNR."""
    scores = {"01": [], "-11": []}
    for i, (lr, hr) in enumerate(loader):
        if i >= max_batches:
            break
        lr01, hr01 = denorm01(lr.to(device)), denorm01(hr.to(device))
        for dom in scores:
            scores[dom].append(psnr01(predict(net, lr01, dom), hr01))
    return max(scores, key=lambda d: float(np.mean(scores[d])))


@torch.no_grad()
def evaluate_split(net, domain, loader, device):
    psnrs, ssims, laps = [], [], []
    t0 = time.perf_counter()
    n_imgs = 0
    for lr, hr in loader:
        lr01, hr01 = denorm01(lr.to(device)), denorm01(hr.to(device))
        sr = predict(net, lr01, domain)
        psnrs.append(psnr01(sr, hr01))
        ssims.append(ssim01(sr, hr01))
        laps.append(lap_var01(sr))
        n_imgs += sr.shape[0]
    dt = time.perf_counter() - t0
    return {
        "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
        "laplacian": float(np.mean(laps)),
        "ms_per_img": dt / max(n_imgs, 1) * 1000.0,
    }


@torch.no_grad()
def bench_latency(net, device, n_iter=30):
    """Forward latency at training crop (48->192) and eval crop (96->384)."""
    out = {}
    for tag, size in (("train", 48), ("eval", 96)):
        x = torch.rand(1, 3, size, size, device=device)
        for _ in range(5):
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            net(x)
        torch.cuda.synchronize()
        out[tag + "_ms"] = (time.perf_counter() - t0) / n_iter * 1000.0
    return out


_INSTALLED_EXTRA = False


def load_spandrel(path):
    """Load any supported upscale checkpoint via spandrel arch autodetect."""
    global _INSTALLED_EXTRA
    from spandrel import ModelLoader
    if not _INSTALLED_EXTRA:
        import spandrel_extra_arches
        spandrel_extra_arches.install()
        _INSTALLED_EXTRA = True
    desc = ModelLoader().load_from_file(str(path))
    if not hasattr(desc, "model"):
        raise TypeError(f"{Path(path).name}: not an image model")
    return desc.model


def main():
    ap = argparse.ArgumentParser(description="benchmark teacher candidates")
    ap.add_argument("--data", default=str(ROOT / "data" / "anime_video_frames"))
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="name=path pairs; default all pth under "
                         "checkpoints/candidates plus incumbent SPAN")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of test images (smoke)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-csv",
                    default=str(ROOT / "results" / "teacher_bench.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    ds = AnimePairDataset(args.data, split="test")
    if args.limit:
        ds.files = ds.files[:args.limit]
    print(f"test split: {len(ds)} images, eval crop {ds.crop_hr}px")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2)

    cands = []
    if args.candidates:
        for spec in args.candidates:
            name, _, path = spec.partition("=")
            cands.append((name or Path(path).stem, Path(path)))
    else:
        if CANDIDATE_DIR.exists():
            cands += [(p.stem, p) for p in sorted(CANDIDATE_DIR.glob("*.pth"))]
        if INCUMBENT_SPAN.exists():
            cands.append(("SPAN_incumbent", INCUMBENT_SPAN))

    rows = []
    for name, path in cands:
        row = {"name": name, "checkpoint": str(path)}
        try:
            if name == "SPAN_incumbent":
                from teacher import SPANTeacher
                t = SPANTeacher(device=device)
                net, domain = t.net, "01"
            else:
                net = load_spandrel(path).to(device).eval()
                domain = probe_domain(net, loader, device)
            n_par = sum(p.numel() for p in net.parameters())
            m = evaluate_split(net, domain, loader, device)
            lat = bench_latency(net, device)
            row.update({
                "status": "ok", "domain": domain,
                "params_M": round(n_par / 1e6, 2),
                "psnr_db": round(m["psnr"], 3),
                "ssim": round(m["ssim"], 4),
                "laplacian": round(m["laplacian"], 6),
                "train_ms": round(lat["train_ms"], 2),
                "eval_ms": round(lat["eval_ms"], 2),
                "split_ms_per_img": round(m["ms_per_img"], 1),
            })
        except Exception as e:              # noqa: BLE001 report and continue
            row["status"] = ("FAIL: " + type(e).__name__ + ": " + str(e))[:200]
        rows.append(row)
        print(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["name", "status", "domain", "params_M", "psnr_db", "ssim",
            "laplacian", "train_ms", "eval_ms", "split_ms_per_img",
            "checkpoint"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")

    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["psnr_db"])
        print("BEST fidelity : %s  %s dB / SSIM %s" % (best["name"], best["psnr_db"], best["ssim"]))
        fastest = min(ok, key=lambda r: r["train_ms"])
        print("FASTEST train : %s  %s ms/crop" % (fastest["name"], fastest["train_ms"]))


if __name__ == "__main__":
    main()
