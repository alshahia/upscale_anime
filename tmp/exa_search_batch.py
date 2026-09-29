import os, json, time, asyncio
import exa_py

exa = exa_py.Exa("0ffdbae1-ddfa-4a06-bb73-0a7aee5d1cb5")

QUERIES = [
    ("anime_sota_2024_2026", "best anime 4x super resolution model 2024 2025 2026 Real-ESRGAN animevideov3 vs Real-CUGAN vs Anime4K vs SUPIR vs SeeSR quality comparison"),
    ("realtime_anime_arch", "lightweight real-time super resolution architecture 2024 2025 RFDN IMDN BSRN ESRT OmniSR ShuffleMixer ELSANet inference latency ms parameters"),
    ("mambair_mamba_sr", "MambaIRv2 state space model super resolution 2024 2025 inference speed quality vs CNN Transformer"),
    ("diffusion_anime_sr", "SUPIR SeeSR DiffBIR StableSR OSEDiff SinSR diffusion super resolution anime perceptual quality latency 2024 2025"),
    ("knowledge_distillation_sr", "knowledge distillation super resolution 2024 2025 feature affinity contrastive spatial curriculum frequency distillation"),
    ("cascade_2x_2x_sr", "2x cascade super resolution 4x scale two-stage lightweight student faster than single 4x model"),
    ("anime_dataset_benchmark", "anime super resolution dataset benchmark 2024 2025 AVC-RealLQ Danbooru ACvc real-world low quality"),
    ("inference_acceleration", "TensorRT ONNX runtime super resolution inference optimization fp16 int8 batch size speedup 2024 2025"),
    ("gan_training_sr", "adversarial training super resolution ESRGAN discriminator 2024 2025 perceptual loss stable training anime line art"),
    ("edge_line_art", "anime line art edge preservation super resolution Sobel edge loss gradient loss structure preservation"),
    ("tta_ensemble_sr", "test time augmentation super resolution flip rotate ensemble 2024 2025 PSNR gain cost"),
    ("perceptual_quality_metric", "MANIQA CLIPIQA LPIPS NIQE perceptual super resolution metric 2024 2025 best perceptual quality anime"),
]

def safe(obj):
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, (list, tuple)): return [safe(x) for x in obj]
    if isinstance(obj, dict): return {str(k): safe(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"): return safe(obj.model_dump())
    if hasattr(obj, "__dict__"): return safe(vars(obj))
    return str(obj)

async def run_one(label, q):
    t0 = time.time()
    try:
        r = exa.search(q, num_results=8, type="auto", contents={"highlights": {"numSentences": 4, "highlightsPerUrl": 3}})
        out = safe({
            "label": label, "query": q,
            "resolved_search_type": r.resolved_search_type,
            "search_time_ms": r.search_time,
            "cost_dollars": r.cost_dollars,
            "results": [
                {"title": x.title, "url": x.url, "published_date": str(getattr(x, "published_date", "") or ""), "author": getattr(x, "author", None), "highlights": (x.highlights or [])[:3]}
                for x in r.results
            ],
        })
        cost = float(getattr(r.cost_dollars, "total", 0) or 0)
        msg = "[" + label + "] " + str(round(time.time()-t0, 1)) + "s cost=" + str(cost) + " hits=" + str(len(r.results))
        print(msg, flush=True)
        return out
    except Exception as exc:
        print("[" + label + "] FAIL " + type(exc).__name__ + ": " + str(exc), flush=True)
        return {"label": label, "query": q, "error": str(exc)}

async def main():
    tasks = [run_one(lbl, q) for lbl, q in QUERIES]
    results = await asyncio.gather(*tasks)
    total = 0.0
    for r in results:
        c = r.get("cost_dollars") or {}
        if isinstance(c, dict):
            total = total + float(c.get("total", 0) or 0)
    out = "E:\\python projects\\upscale_anime\\tmp\\exa_search_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("ALL DONE. " + str(len(results)) + " queries, cost=" + str(total), flush=True)
    print("Saved: " + out, flush=True)

asyncio.run(main())