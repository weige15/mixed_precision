# Open-Source Codebase Filter: Soft Pruning / Adaptive Mixed Precision

This list keeps papers that are relevant to soft-pruning-style mixed precision
and have a public codebase or implementation repository.

## Best Picks

| Priority | Paper / project | Venue or review signal | Codebase | Why it matters |
|---|---|---|---|---|
| 1 | SliM-LLM: Salience-Driven Mixed-Precision Quantization for LLMs | ICML 2025 | https://github.com/Aaronhuang-778/SliM-LLM | Best open-source LLM-specific salience-to-bitwidth method. Includes group-wise bitwidth allocation and mixed-precision packing/deployment notes. |
| 2 | Bayesian Bits: Unifying Quantization and Pruning | NeurIPS 2020 | https://github.com/Qualcomm-AI-research/BayesianBits | Best open-source conceptual bridge: learn stochastic gates over residual bits, with 0-bit as pruning. |
| 3 | HAQ: Hardware-Aware Automated Quantization with Mixed Precision | CVPR 2019 oral | https://github.com/mit-han-lab/haq | Best open-source hardware-aware bitwidth assignment baseline. |
| 4 | ChanMix: Channel-Aware Mixed-Precision Quantization for Efficient Long-Context Inference | ICLR 2026 | https://github.com/cxiliao/ChanMix | Newer OpenReview/ICLR codebase for channel-aware mixed precision in long-context LLM inference. |
| 5 | Instance-Aware Dynamic Neural Network Quantization | CVPR 2022 | https://github.com/huawei-noah/Efficient-Computing | Best reviewed open-source precedent for input-conditioned dynamic bitwidth selection, although vision-focused. |
| 6 | DiffQ: Differentiable Model Compression via Pseudo Quantization Noise | Paper + PyPI/GitHub implementation | https://github.com/facebookresearch/diffq | Strong open-source implementation for differentiable bit allocation per weight or group. Useful even if not LLM-specific. |
| 7 | MatGPTQ: Accurate and Efficient Post-Training Matryoshka Quantization | 2026 preprint | https://github.com/IST-DASLab/MatGPTQ | Open-source follow-up to Matryoshka Quantization with bit-slicing, mixed-precision execution, and kernels. Use as code-backed storage reference rather than as the original ICML MatQuant codebase. |
| 8 | Any-Precision LLM | ICML 2024 oral | https://github.com/SNU-ARC/any-precision-llm | Strong open-source multi-bitwidth storage/serving baseline. |
| 9 | DP-LLM | NeurIPS 2025 | https://github.com/SNU-ARC/DP-LLM | Runtime input-conditioned layer-wise precision selection. |
| 10 | QuEPT | AAAI 2026 | https://github.com/xuke225/QuEPT | Elastic precision Transformers with one-shot calibration and multi-bit switching. |
| 11 | QAQ: Quality Adaptive Quantization for LLM KV Cache | 2024 preprint / ICCV workshop | https://github.com/ClubieDong/QAQ-KVCacheQuantization | Cache-side adaptive precision with outlier and attention-aware protection. |
| 12 | ARKV | 2026 preprint | https://github.com/Large-scale-Sustainable-Computing-LSC/ARKV | Tri-state full-precision, low-precision, or evicted KV cache tokens. |
| 13 | NestedFP | NeurIPS 2025 | https://github.com/SNU-ARC/NestedFP | Memory-efficient FP16/FP8 nested representation with serving kernels. |

## Good Ideas But No Public Code Found In This Pass

| Paper | Status |
|---|---|
| FracBits | Strong AAAI 2021 paper, but no official public codebase found in the current search. |
| QAQ: Query-adaptive Mixed-precision Quantization for LLMs | Very relevant to prompt-conditioned bit-plane routing, but no codebase found from the paper/OpenReview/NeurIPS pages in this pass. |
| AdaQuantLM | OpenReview workshop paper; no codebase found in this pass. |
| Prompt-Adaptive Quantization | OpenReview workshop paper; no codebase found in this pass. |
| ScaleBITS | Relevant arXiv 2026 preprint; no public codebase found in this pass. |
| MoBiQuant | Highly relevant arXiv 2026 token-adaptive residual-bit-slice method; no public codebase found in this pass. |
| Don't Waste Bits | Highly relevant adaptive per-token KV-cache bitwidth paper; no public codebase found in this pass. |
| FineQ / FGMP / RAMP / CMPQ | Relevant systems or allocation papers, but no public codebase found in this pass. |

## Recommended Reproducible Stack

For a reproducible project, start with:

1. **SliM-LLM** for LLM salience-driven group-wise precision.
2. **Bayesian Bits** for the soft-pruning-to-bit-allocation mechanism.
3. **HAQ** for hardware-aware assignment.
4. **ChanMix** or **MatGPTQ** for newer LLM-oriented storage/layout work.
5. **Instance-Aware Dynamic Quantization** if the project needs an
   input-conditioned controller baseline.
