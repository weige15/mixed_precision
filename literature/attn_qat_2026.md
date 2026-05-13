# Attn-QAT: 4-Bit Attention With Quantization-Aware Training

- **Authors:** Peiyuan Zhang, Matthew Noto, Wenxuan Tan, Chengquan Jiang, Will Lin, Wei Zhou, Hao Zhang
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2603.00040
- **DOI:** https://doi.org/10.48550/arXiv.2603.00040

## Summary

Attn-QAT studies 4-bit attention with quantization-aware training. Its central claim is that attention is a major obstacle to end-to-end FP4 execution because attention activations are heavy-tailed and FP4 has a tiny dynamic range. The paper reports that naive "drop-in" QAT can be unstable when the forward pass is low precision but the backward path retains hidden high-precision assumptions.

This is highly relevant to precision-island design: it suggests attention should be treated as a high-risk operation for aggressive low precision unless the forward and backward numerics are designed together.

## Relevance To This Project

For H6, attention softmax and attention-score recomputation should be classified as conservative candidates. The project can still measure attention activation outliers and loss deltas, but should avoid treating 4-bit attention as a simple dtype toggle during LoRA training.

## Key Takeaways

- Attention is a likely precision-sensitive region under FP4/subbyte training.
- Hidden precision assumptions in fused kernels can destabilize training.
- Backward-pass numerical consistency can matter as much as forward quantization.

## Evidence Gaps

- The work is very recent and specialized to QAT/kernel design.
- It is not a BF16 LoRA fine-tuning study.
- It does not provide a cheap CPU/GPU-agnostic policy for ordinary PyTorch experiments.
