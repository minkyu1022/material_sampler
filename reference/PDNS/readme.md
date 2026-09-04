# Official Implementation for *Proximal Diffusion Neural Sampler* (ICLR 2026)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2510.03824)
[![Paper](https://img.shields.io/badge/Paper-ICLR_2026-b31b1b)](https://openreview.net/forum?id=XTHQqS7ObC)

Thanks for your interest in our work. We are currently organizing the code for our paper, and release it before the conference in April 2026. Please stay tuned!

---

This repository contains the official implementation of the Proximal Diffusion Neural Sampler (PDNS). PDNS addresses the instability of learning a diffusion-based neural sampler when the target distribution is multimodal with significant barriers separating the modes by tackling the stochastic optimal control problem via proximal point method on the space of path measures.

Please check the `discrete` folder for the code of discrete sampling and the `continuous` folder for the code of continuous sampling.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{guo2026proximal,
    title     = {Proximal Diffusion Neural Sampler},
    author    = {Wei Guo and Jaemoo Choi and Yuchen Zhu and Molei Tao and Yongxin Chen},
    booktitle = {The Fourteenth International Conference on Learning Representations},
    year      = {2026},
    url       = {https://openreview.net/forum?id=XTHQqS7ObC}
}
```

## Acknowledgements

This repository is developed based on [MDNS](https://github.com/yuchen-zhu-zyc/MDNS) under the MIT License.
