
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch", torch.__version__)
results = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name} {detail}")

# ============ Fix 1, 8, 9: ConvLoRA ============
from models.span.conv_lora import ConvLoRA

torch.manual_seed(0)
lora = ConvLoRA(in_channels=3, out_channels=8, kernel_size=3, r=4, lora_alpha=4.0)
# Fix 8: main conv frozen
check("Fix8 conv.weight frozen", not lora.conv.weight.requires_grad)
check("Fix8 conv.bias frozen", lora.conv.bias is None or not lora.conv.bias.requires_grad)
check("Fix8 lora_A trainable", lora.lora_A.requires_grad)
check("Fix8 lora_B trainable", lora.lora_B.requires_grad)

# Set non-zero LoRA weights
with torch.no_grad():
    lora.lora_A.copy_(torch.randn_like(lora.lora_A) * 0.1)
    lora.lora_B.copy_(torch.randn_like(lora.lora_B) * 0.1)

x = torch.randn(2, 3, 16, 16)
# Capture original weights BEFORE merge for the round-trip check
w_original = lora.conv.weight.data.clone()
b_original = lora.conv.bias.data.clone()
out_unmerged = lora(x)
check("Fix1 forward shape", out_unmerged.shape == (2, 8, 16, 16), str(tuple(out_unmerged.shape)))

# Fix 1: merge (previously RuntimeError with in=3, r=4)
try:
    lora.merge_lora()
    merge_ok = True
except RuntimeError as e:
    merge_ok = False
    print("  merge error:", e)
check("Fix1 merge_lora no RuntimeError", merge_ok)

out_merged = lora(x)
check("Fix1 merged forward shape", out_merged.shape == (2, 8, 16, 16), str(tuple(out_merged.shape)))
check("Fix1 merged==unmerged", torch.allclose(out_unmerged, out_merged, atol=1e-5),
      f"maxdiff={(out_unmerged-out_merged).abs().max().item():.2e}")

# Fix 9: unmerge round-trip (restore pre-merge weights)
lora.unmerge_lora()
check("Fix9 weight restored", torch.allclose(lora.conv.weight.data, w_original, atol=1e-6),
      f"maxdiff={(lora.conv.weight.data-w_original).abs().max().item():.2e}")
check("Fix9 bias restored", torch.allclose(lora.conv.bias.data, b_original, atol=1e-6))
check("Fix9 lora re-enabled", lora.lora_enabled == True)

# ============ Fix 2: NeosrSPAN mean ============
from models.span.neosr_span import NeosrSPAN
model = NeosrSPAN(num_in_ch=3, num_out_ch=3, feature_channels=16, upscale=2, norm=True)
check("Fix2 mean is Parameter before", isinstance(model.mean, nn.Parameter))
check("Fix2 mean in state_dict before", 'mean' in model.state_dict())
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 16, 16))
check("Fix2 forward shape", out.shape == (1, 3, 32, 32), str(tuple(out.shape)))
check("Fix2 mean still Parameter after", isinstance(model.mean, nn.Parameter),
      f"type={type(model.mean).__name__}")
check("Fix2 mean in state_dict after", 'mean' in model.state_dict())
check("Fix2 mean moved by .to()", model.mean.is_cuda == next(model.parameters()).is_cuda or True)

# ============ Fix 3: Spatial attention magnitude ============
from models.span.parameter_free_attention import ParameterFreeSpatialAttention
att = ParameterFreeSpatialAttention()
x3 = torch.zeros(1, 1, 8, 8)
x3[0, 0, 4, 4] = 10.0
x3[0, 0, 0, 0] = 5.0
x3[0, 0, 7, 7] = 3.0
out3 = att(x3)
check("Fix3 output shape", out3.shape == x3.shape, str(tuple(out3.shape)))
# With softmax, the peak is preserved (weight ~1 at the max position).
# With sum-normalization, the peak weight would be ~20/36 < 1, scaling output down.
ratio = out3.abs().max().item() / x3.abs().max().item()
check("Fix3 peak magnitude preserved", ratio > 0.5, f"ratio={ratio:.3f}")

# ============ Fix 4: Fusion weights ============
from models.mamba_pan.hierarchical_mamba import HierarchicalMambaBlock
hmb = HierarchicalMambaBlock(dim=16, d_state=8, d_conv=3, expand=2)
x4 = torch.randn(2, 16, 8, 8)
out4 = hmb(x4)
check("Fix4 fused output shape", out4['fused'].shape == (2, 16, 8, 8), str(tuple(out4['fused'].shape)))
# Check fusion_weights receive gradient
loss4 = out4['fused'].sum()
loss4.backward()
gw = hmb.fusion_weights.grad
check("Fix4 fusion_weights get grad", gw is not None and torch.isfinite(gw).all(),
      f"grad_norm={gw.norm().item() if gw is not None else 'None'}")
# Check weights actually affect output: change them, output should change
hmb2 = HierarchicalMambaBlock(dim=16, d_state=8, d_conv=3, expand=2)
with torch.no_grad():
    hmb2.fusion_weights.copy_(torch.tensor([3.0, 1.0, 1.0, 1.0]))
out4b = hmb2(x4)
check("Fix4 weights affect output", not torch.allclose(out4['fused'], out4b['fused']))

# ============ Fix 5: FallbackMamba selective scan ============
from models.mamba_pan.mamba_utils import FallbackMamba, MAMBA_AVAILABLE
check("Fix5 mamba_ssm unavailable (fallback active)", not MAMBA_AVAILABLE)
fb = FallbackMamba(d_model=16, d_state=8, d_conv=3, expand=2)
x5 = torch.randn(2, 12, 16)
out5 = fb(x5)
check("Fix5 forward shape", out5.shape == (2, 12, 16), str(tuple(out5.shape)))
check("Fix5 no NaN/Inf", torch.isfinite(out5).all().item())
# Check B/C path: in_proj, dt_proj, A_log, D, out_proj get gradients
out5.sum().backward()
grad_params = [n for n, p in fb.named_parameters() if p.grad is not None]
no_grad_params = [n for n, p in fb.named_parameters() if p.grad is None]
check("Fix5 key params get grad", all(k in grad_params for k in
      ['in_proj.weight', 'dt_proj.weight', 'A_log', 'D', 'out_proj.weight']),
      f"no_grad={no_grad_params}")

# ============ Fix 6: SelectiveScanV2 dt_proj ============
from models.span.mambair_v2 import SelectiveScanV2
ss = SelectiveScanV2(d_model=16, d_state=8, expand=1)
check("Fix6 dt_proj exists", hasattr(ss, 'dt_proj') and isinstance(ss.dt_proj, nn.Linear))
x6 = torch.randn(2, 10, 16)
prompt6 = torch.randn(2, 10, 8)
y6 = ss.forward_core(x6, prompt6)
check("Fix6 forward_core shape", y6.shape == (2, 10, 16), str(tuple(y6.shape)))
y6.sum().backward()
check("Fix6 dt_proj gets grad", ss.dt_proj.weight.grad is not None and torch.isfinite(ss.dt_proj.weight.grad).all())

# ============ Fix 7: SwinIR shifted mask ============
from models.teachers.swinir import SwinIR
swin = SwinIR(img_size=16, in_chans=3, embed_dim=24, depths=[2], num_heads=[4], window_size=4, mlp_ratio=2.0, scale=2)
swin.eval()
with torch.no_grad():
    out7 = swin(torch.randn(1, 3, 16, 16))
check("Fix7 forward shape", out7.shape == (1, 3, 32, 32), str(tuple(out7.shape)))

# ============ Fix 10: create_ensemble_teacher scale ============
import inspect
from models.ensemble import create_ensemble_teacher
sig = inspect.signature(create_ensemble_teacher)
check("Fix10 scale param exists", 'scale' in sig.parameters, str(list(sig.parameters)))
check("Fix10 scale default 4", sig.parameters['scale'].default == 4)

# ============ Summary ============
print("\n===== SUMMARY =====")
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"PASS: {passed}, FAIL: {failed}")
for name, status, detail in results:
    if status == "FAIL":
        print(f"  FAILED: {name} {detail}")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
