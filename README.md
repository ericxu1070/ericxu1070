### Hi, I'm Eric

I'm doing a one-year M.S. in Scientific Computing and Applied Mathematics at UC Santa Cruz, where I also got my B.S. in applied math. I work in Ashesh Chattopadhyay's TACS Lab on forecasting extreme weather a few weeks ahead, and part-time at Vayuh.ai on the same problem. I'm applying to PhD programs for Fall 2027.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/aires-dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="assets/aires-light.gif">
  <img alt="Rare-event sampling forecast of the June 2021 Pacific Northwest heat dome: 64 FourCastNet3 walkers are cloned and killed every 3 days, and most end up reaching the observed heat" src="assets/aires-light.gif" width="100%">
</picture>

A run of my AI+RES pipeline: 64 FourCastNet3 walkers start on June 7, 2021, three weeks before the Pacific Northwest heat dome. Every 3 days, FourCastNet3 forecasts score each walker on how hot June 22-28 will be, and the best-scored walkers are cloned while the rest are killed ([Lancelin et al., 2025](https://arxiv.org/abs/2510.27066)). 36 of 64 walkers reach the observed week (+7.7 °C). A plain 24-member ensemble from the same start gets 1.

**Projects**
- [DeepSEA-Transfer-Learning](https://github.com/ericxu1070/DeepSEA-Transfer-Learning): fine-tuning DeepSEA, a genomic model trained on human data, on mouse data
- [council-of-sumer](https://github.com/ericxu1070/council-of-sumer): LLM advisors argue and vote on how to run an ancient city (UCSC AI Hackathon)
- [invobs-data-assimilation](https://github.com/ericxu1070/invobs-data-assimilation): my PyTorch reimplementation of Frerix et al. (ICML 2021), 4D-Var with a learned inverse observation operator

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![JAX](https://img.shields.io/badge/JAX-5E97F6?logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-GPU%20kernels-76B900) ![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white) ![HPC](https://img.shields.io/badge/HPC-SLURM%20%7C%20PBS%20%7C%20Derecho-555555)

[Website](https://ericxu1070.github.io/) · [Resume](https://ericxu1070.github.io/resume.pdf) · [LinkedIn](https://linkedin.com/in/exu06) · ericxu1070 [at] gmail [dot] com
