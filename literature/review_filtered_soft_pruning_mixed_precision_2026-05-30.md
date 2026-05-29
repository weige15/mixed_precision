# Review-Filtered Selection: Soft Pruning / Adaptive Mixed Precision

This file filters the collected papers by review confidence. "Good reviews" is
interpreted conservatively as accepted at a strong peer-reviewed venue or a
clearly accepted OpenReview venue. Some workshop papers are still useful because
they match the prompt-conditioned idea closely, but they should not be the main
evidence unless later accepted at a stronger venue.

## Best Core Set

| Priority | Paper | Review signal | Why keep it |
|---|---|---|---|
| 1 | SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models | ICML 2025 poster on OpenReview | Best reviewed LLM-specific salience-to-bitwidth paper. It directly supports assigning more bits to important weight groups while keeping the layout hardware-friendly. |
| 2 | Matryoshka Quantization | ICML 2025 / PMLR | Best reviewed storage-efficient multi-precision paper. It supports the "store once, serve many bitwidths" answer via nested bit planes. |
| 3 | Bayesian Bits: Unifying Quantization and Pruning | NeurIPS 2020 accepted, public reviews/metareview | Best conceptual bridge from soft pruning to bit allocation: 0-bit is pruning, extra bits are retained information. |
| 4 | HAQ: Hardware-Aware Automated Quantization with Mixed Precision | CVPR 2019 | Best classic hardware-aware precision assignment paper. It grounds the hardware-cost side of the proposal. |
| 5 | FracBits: Mixed Precision Quantization via Fractional Bit-Widths | AAAI 2021 | Strong reviewed reference for soft/fractional bitwidth during optimization followed by deployable hard assignment. |
| 6 | Instance-Aware Dynamic Neural Network Quantization | CVPR 2022 | Best reviewed precedent that input instances can choose different bitwidth policies, even though it is vision-focused rather than LLM-specific. |

## Useful But Lower Review Confidence

| Paper | Review signal | How to use |
|---|---|---|
| QAQ: Query-adaptive Mixed-precision Quantization for LLMs | NeurIPS 2025 ML for Systems workshop on OpenReview | Highly relevant to prompt-conditioned bit-plane loading. Use as emerging prior, not as the strongest evidence. |
| AdaQuantLM: LLM Quantization with Adaptive Bit-Widths | NeurIPS 2024 Compression Workshop on OpenReview | Useful for additive-codeword storage, but workshop-level and narrower experimental evidence. |
| Prompt-Adaptive Quantization | AAAI 2026 AIR-FM workshop on OpenReview | Good baseline for prompt routing, but it routes among pre-quantized variants, so it does not solve compact single-representation storage. |
| ScaleBITS | arXiv 2026 preprint | Very relevant hardware-aligned LLM bit allocation, but no peer-review signal found yet. Use cautiously as recent context. |
| MoBiQuant | arXiv 2026 preprint | Highly related to token-adaptive precision and residual bit-slice storage, but no strong peer-review signal or codebase found yet. |
| FineQ | arXiv 2025 / DATE 2025 comment | Strong systems idea for aligned memory access, but review metadata was not as directly visible as the core set. |
| FGMP | arXiv 2025 preprint | Good hardware/software co-design reference for block precision and activation routing, but no strong review signal found yet. |
| RAMP | arXiv 2026 preprint | Good learned-policy framing, but preprint only. |
| CMPQ | arXiv 2024/2025 preprint | Useful channel-wise allocation idea, but no strong review signal found yet. |

## Do Not Use As Main Evidence

| Paper | Reason |
|---|---|
| AutoMixQ | OpenReview page indicates withdrawn ICLR 2026 submission. |
| SAMPQ | OpenReview page indicates withdrawn ICLR 2026 submission. |

## Recommended Citation Strategy

Use this hierarchy in a proposal or paper:

1. Lead with **SliM-LLM**, **Matryoshka Quantization**, and **Bayesian Bits**.
2. Use **HAQ** and **FracBits** to show the idea has a mature reviewed history.
3. Use **Instance-Aware Dynamic Quantization** to justify input-conditioned
   precision decisions.
4. Mention **QAQ** only as the most directly aligned emerging LLM prompt/query
   method.
5. Mention **ScaleBITS/FineQ/FGMP** as systems context if discussing storage,
   packing, alignment, and kernels.

For the open-source-only subset, see
[Open-Source Codebase Filter: Soft Pruning / Adaptive Mixed Precision](open_source_soft_pruning_mixed_precision_2026-05-30.md).

## Selected Research Direction After Filtering

The strongest defensible proposal is:

```text
Learn or calibrate a structured importance decomposition, store weights in
nested/grouped bit planes, and let a prompt/hardware-aware policy decide how
many planes to use.
```

This builds on reviewed work while still leaving a clear gap:

- SliM-LLM handles salience-to-group bitwidth.
- Matryoshka handles nested precision storage.
- Bayesian Bits handles soft pruning as bit allocation.
- HAQ handles hardware-aware assignment.
- The remaining gap is a unified method that combines prompt difficulty,
  hardware cost, and storage-efficient nested/disentangled weight information.
