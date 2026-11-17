# README media assets

Committed images used by the top-level <a href="../../README.md">README.md</a> and its <a href="../../README.ar.md">Arabic translation</a>. Every file is static and self-contained: nothing here is hotlinked or loaded from a third-party host.

## Files

| File | Dimensions | Bytes | SHA-256 | Used by |
|---|---|---:|---|---|
| `banner.png` | 800x600 | 19,478 | `a0dd632d3ccc02815187d17a7ce9ed50f77c9fffb2c76b7ea95186f0f7649b98` | Title banner |
| `example-lr.jpg` | 960x540 | 90,471 | `26d681630141535bfcf237db02cfa3c8ae26c12ae63fcd20aec5a0a3fdf0b40a` | Before/after input; detail comparison source |
| `example-sr.jpg` | 1600x900 | 193,875 | `a7abe5b334266546505c0e29bf143a32c1419b3e7938a86b1c5d5e903627ef5a` | Before/after 4x output |
| `example-detail.png` | 1283x390 | 352,828 | `62f0a231711c6a7387766ed59203cad8ce8a762adbdb57b785c9ef71c1078348` | Detail comparison |
| `gui-screenshot.png` | 1200x311 | 51,734 | `891bf729c4cb60bf1df2f123075d43744a75fe79a009ca31b3552f63d91caf2d` | GUI screenshot |

Total: **708,386 bytes (0.68 MB)**, below the 2 MB documentation-media budget.

## Demo pair provenance

- Input: <a href="../../assets/examples/lr.png"><code>assets/examples/lr.png</code></a>, the repository's own example asset (1280x720 landscape frame).
- 4x output: produced from that input with the shipped 0.2.0-preview inference stack:
  - checkpoint: `pretrained/realesr-animevideov3.pth` - a **demo checkpoint**, not a released model and not committed to this repository (it is downloaded or imported through the GUI's **Models** tab);
  - architecture kind: `srvgg`;
  - scale: 4x;
  - inference: tiled, tile size 256 px with 8 px overlap;
  - output: 5120x2880 RGB PNG (kept in the internal review workspace, not committed).
- Generated 2026-11-16; committed to `docs/media/` on 2026-11-17.
- `example-lr.jpg`: the input downscaled to 960x540 (JPEG, quality 85, progressive).
- `example-sr.jpg`: the 4x output downscaled to 1600x900 (JPEG, quality 85, progressive) for the web.
- `example-detail.png`: two 640x360 panels plus a 30 px label strip and a thin divider. Both panels show the same 320x180 input region at (100, 280): the left panel is the input enlarged 2x with nearest-neighbour, the right panel is the corresponding 4x output region downscaled to 640x360.

These are project demo images, not a benchmark comparison. Benchmark numbers quoted in the README remain project-reported measurements linked to their source documents; these images do not establish a quality claim.

### Legacy asset

<a href="../../assets/examples/output.png"><code>assets/examples/output.png</code></a> is a historical demo output from the initial 2026-06-14 commit, produced by an older run. It is visibly softer and colour-shifted, and it is **not** used by either README. It stays in the repository only as historical content.

## GUI screenshot provenance

`gui-screenshot.png` is a real window capture of `python -m apps.anime_upscaler_gui --no-splash` from this development checkout (`anime-upscaler-gui 0.2.0`):

- captured with the Win32 `PrintWindow` API (`PW_RENDERFULLCONTENT`) against the app window only - no desktop capture, no compositing, no mocked or edited UI content;
- cropped to the app's **Models** tab; the preview/status/log area below the tab is not included in the image;
- the state shown (installed checkpoint list, selected row, preset catalog) is real installed-model metadata from the development environment, reached through the app's own controls.

Checkpoint weights are not part of the repository, so the installed list can differ between machines.

## Banner provenance

`banner.png` is derived from the GUI's bundled splash artwork, <a href="../../apps/anime_upscaler_gui/anime_upscaler_gui/assets/splash/hero-800x600.png"><code>hero-800x600.png</code></a>. It is converted to RGB and quantised to a 256-colour palette to keep the repository light; the 800x600 dimensions are preserved.

## Third-party content

The demo input is the repository's own example asset. No third-party anime frames or third-party output images are included, and no image is loaded from an external host.
