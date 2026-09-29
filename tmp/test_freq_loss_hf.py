
import torch
import torch.nn.functional as F

def _freq_magnitude_loss_hf(s_out, hr, cutoff_frac=0.25):
    a = s_out.clamp(0, 1)
    b = hr.clamp(0, 1)
    A = torch.fft.fftshift(torch.fft.fft2(a, norm="ortho"), dim=(-2, -1))
    B = torch.fft.fftshift(torch.fft.fft2(b, norm="ortho"), dim=(-2, -1))
    mag_a = torch.abs(A)
    mag_b = torch.abs(B)
    _, _, H, W = mag_a.shape
    yy = torch.arange(H, device=mag_a.device).float() - (H - 1) / 2.0
    xx = torch.arange(W, device=mag_a.device).float() - (W - 1) / 2.0
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    r_max = rr.max().clamp(min=1.0)
    hf_mask = (rr > cutoff_frac * r_max).to(mag_a.dtype)
    return (mag_a * hf_mask - mag_b * hf_mask).abs().mean()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Better LF signal: a smooth 2D Gaussian (low-frequency content)
H, W = 192, 192
yy = torch.arange(H, device=device).float() - (H - 1) / 2.0
xx = torch.arange(W, device=device).float() - (W - 1) / 2.0
yy, xx = torch.meshgrid(yy, xx, indexing="ij")
lf_signal = torch.exp(-(yy ** 2 + xx ** 2) / (2 * 30 ** 2))  # sigma=30 -> LF band
lf_signal = (lf_signal / lf_signal.max()) * 0.3  # scale to 0..0.3

# HF signal: high-frequency checkerboard (all energy at Nyquist)
hf_signal = torch.zeros(H, W, device=device)
hf_signal[::2, ::2] = 0.3  # checkerboard

hr = torch.rand(2, 3, H, W, device=device) * 0.5 + 0.25

# LF-only diff
s_lf = (hr + lf_signal.unsqueeze(0).unsqueeze(0)).clamp(0, 1)
loss_lf = _freq_magnitude_loss_hf(s_lf, hr, cutoff_frac=0.25)

# HF-only diff
s_hf = (hr + hf_signal.unsqueeze(0).unsqueeze(0)).clamp(0, 1)
loss_hf = _freq_magnitude_loss_hf(s_hf, hr, cutoff_frac=0.25)

# Same-energy diff: pure noise
s_noise = (hr + torch.randn_like(hr) * 0.1).clamp(0, 1)
loss_noise = _freq_magnitude_loss_hf(s_noise, hr, cutoff_frac=0.25)

# Identity
loss_id = _freq_magnitude_loss_hf(hr, hr, cutoff_frac=0.25)

# Print also: total energy of LF/HF signals in their respective bands
lf_spectrum = torch.abs(torch.fft.fftshift(torch.fft.fft2(lf_signal.unsqueeze(0).unsqueeze(0), norm="ortho")))
hf_spectrum = torch.abs(torch.fft.fftshift(torch.fft.fft2(hf_signal.unsqueeze(0).unsqueeze(0), norm="ortho")))
_, _, H2, W2 = lf_spectrum.shape
yy2 = torch.arange(H2, device=device).float() - (H2 - 1) / 2.0
xx2 = torch.arange(W2, device=device).float() - (W2 - 1) / 2.0
yy2, xx2 = torch.meshgrid(yy2, xx2, indexing="ij")
rr2 = torch.sqrt(yy2 ** 2 + xx2 ** 2)
r_max2 = rr2.max()
hf_mask = (rr2 > 0.25 * r_max2)
lf_mask = ~hf_mask

lf_energy_in_hf = (lf_spectrum * hf_mask.float()).mean().item()
lf_energy_in_lf = (lf_spectrum * lf_mask.float()).mean().item()
hf_energy_in_hf = (hf_spectrum * hf_mask.float()).mean().item()
hf_energy_in_lf = (hf_spectrum * lf_mask.float()).mean().item()

print(f"LF signal energy in LF band: {lf_energy_in_lf:.6f}")
print(f"LF signal energy in HF band: {lf_energy_in_hf:.6f}  (should be very small)")
print(f"HF signal energy in LF band: {hf_energy_in_lf:.6f}  (should be 0)")
print(f"HF signal energy in HF band: {hf_energy_in_hf:.6f}")
print()
print(f"identity:  {loss_id.item():.6e}")
print(f"LF-only:   {loss_lf.item():.6f}  (should be << HF-only)")
print(f"HF-only:   {loss_hf.item():.6f}  (should be >> LF-only)")
print(f"random:    {loss_noise.item():.6f}")
print()
print(f"ratio LF/HF: {loss_lf.item() / loss_hf.item():.4f}  (should be < 0.1 if mask works)")
