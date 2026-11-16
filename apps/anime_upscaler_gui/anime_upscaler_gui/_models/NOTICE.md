# Vendored model code notice

`anime_upscaler_gui/_models/` is generated from `src/anime_sr/models/` by
`scripts/sync_vendored_models.py`; do not hand-edit it. The GUI wheel bundles
this torch-only code closure so the app runs without the `anime_sr` package.

The vendored code is derived from, or implements architectures described by,
these upstream projects:

| Vendored file | Upstream project / reference |
|---|---|
| `span/neosr_span.py` | neosr-project/neosr -- `neosr/archs/span_arch.py` |
| `span/mambair_v2.py` | csguoh/MambaIR -- `mambairv2_arch.py` (MambaIRv2, CVPR 2025) |
| `students/mambair.py` | MambaIRv2-style visual state-space student; Mamba paper (Gu and Dao, 2023) |
| `eranet.py` | ERANet (NevermindNilas) |
| `msrsr.py` | TencentARC MSRSWVSR (AnimeSR) |
| `students/__init__.py` | RFDN-style student (Sun et al., AIM 2020) |
| `teachers/spandrel.py` | spandrel checkpoint-loading library |
| `inference/tta.py` | In-project D4 test-time augmentation code (no upstream project cited in the source) |

Notes:

- No copyright or SPDX license header was present in the upstream source files
  when they were vendored (checked 2026-11-16), so the repository MIT license
  covers only the in-project modifications.
- Optional integrations (`mamba-ssm` CUDA kernels, `spandrel`) are imported
  lazily and are not bundled in the wheel.
- **Licensing and attribution review is required before any public release.**
  Each upstream project lists its own license/terms; verify them before
  distributing a wheel that contains this code. This file is not a license and
  grants no rights.
