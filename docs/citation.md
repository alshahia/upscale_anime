# Citation

If you use this software, its models, or the distilled students in academic or technical work, please cite the software release and the relevant upstream papers.

## Cite this software

Citation metadata lives in [CITATION.cff](https://github.com/alshahia/upscale_anime/blob/release/gui-preview/CITATION.cff) (CFF 1.2.0). GitHub's **Cite this repository** button is powered by this file on branches where it is present.

```bibtex
@misc{alshahia2026animeupscaler,
  author       = {alshahia},
  title        = {Anime Upscaler GUI \& anime-sr: anime super-resolution toolkit},
  year         = {2026},
  version      = {0.2.0-preview},
  howpublished = {GitHub pre-release},
  url          = {https://github.com/alshahia/upscale_anime/releases/tag/v0.2.0-preview},
  note         = {Pre-release software; version 0.2.0-preview}
}
```

## Key papers

The architectures and training methods in this repository build on the following work. The entries are provided for convenience; please check the original papers for the definitive metadata.

### SPAN — Swift Parameter-free Attention Network (teacher architecture)

```bibtex
@misc{wan2023span,
  title         = {Swift Parameter-free Attention Network for Efficient Super-Resolution},
  author        = {Wan, Cheng and Yu, Hongyuan and Li, Zhiqi and Chen, Yihang and Zou, Yajun and Liu, Yuqing and Yin, Xuanwu and Zuo, Kunlong},
  year          = {2023},
  eprint        = {2311.12770},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2311.12770}
}
```

### MambaIR — state-space image restoration baseline

```bibtex
@misc{guo2024mambair,
  title         = {MambaIR: A Simple Baseline for Image Restoration with State-Space Model},
  author        = {Guo, Hang and Li, Jinmin and Dai, Tao and Ouyang, Zhihao and Ren, Xudong and Xia, Shu-Tao},
  year          = {2024},
  eprint        = {2402.15648},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2402.15648}
}
```

### MambaIRv2 — attentive state-space restoration

```bibtex
@misc{guo2024mambairv2,
  title         = {MambaIRv2: Attentive State Space Restoration},
  author        = {Guo, Hang and Guo, Yong and Zha, Yaohua and Zhang, Yulun and Li, Wenbo and Dai, Tao and Xia, Shu-Tao and Li, Yawei},
  year          = {2024},
  eprint        = {2411.15269},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2411.15269}
}
```

### RFDN — Residual Feature Distillation Network (student architecture)

```bibtex
@misc{liu2020rfdn,
  title         = {Residual Feature Distillation Network for Lightweight Image Super-Resolution},
  author        = {Liu, Jie and Tang, Jie and Wu, Gangshan},
  year          = {2020},
  eprint        = {2009.11551},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2009.11551}
}
```

### Real-ESRGAN — degradation modeling and SRVGG-family models

```bibtex
@misc{wang2021realesrgan,
  title         = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author        = {Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  year          = {2021},
  eprint        = {2107.10833},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2107.10833}
}
```

### AnimeSR — real-world animation video super-resolution

```bibtex
@misc{wu2022animesr,
  title         = {AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos},
  author        = {Wu, Yanze and Wang, Xintao and Li, Gen and Shan, Ying},
  year          = {2022},
  eprint        = {2206.07038},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2206.07038}
}
```

### APISR — anime-production-inspired degradation and restoration

```bibtex
@misc{wang2024apisr,
  title         = {APISR: Anime Production Inspired Real-World Anime Super-Resolution},
  author        = {Wang, Boyang and Yang, Fengyu and Yu, Xihang and Zhang, Chao and Zhao, Hanbin},
  year          = {2024},
  eprint        = {2403.01598},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2403.01598}
}
```

## Pretrained checkpoints and model code

- Pretrained SPAN warm-start checkpoints come from [Phhofm/models](https://github.com/Phhofm/models); respect their licenses when redistributing checkpoints.
- The GUI's inference paths and model/checkpoint interoperability build on [spandrel](https://github.com/chaiNNer-org/spandrel), [neosr](https://github.com/muslll/neosr), [BasicSR](https://github.com/XPixelGroup/BasicSR) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
- Vendored model code retains its upstream licenses; the full attribution list is in [apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md](https://github.com/alshahia/upscale_anime/blob/release/gui-preview/apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md).

## License

This repository is released under the MIT License (see the [LICENSE](https://github.com/alshahia/upscale_anime/blob/release/gui-preview/LICENSE) file). Upstream model code and checkpoints remain under their own licenses.
