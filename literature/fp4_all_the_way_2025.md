# FP4 All the Way: Fully Quantized Training of LLMs

- **Authors:** Brian Chmiel, Maxim Fishman, Ron Banner, Daniel Soudry
- **Year:** 2025
- **Source:** https://arxiv.org/abs/2505.19115
- **DOI:** https://doi.org/10.48550/arXiv.2505.19115

## Summary

FP4 All the Way studies fully quantized LLM training with predominantly 4-bit floating point precision for weights, activations, and gradients. It reports that FP4 training can approach BF16 baselines when design details such as block size, scaling format, and rounding are chosen carefully. The paper emphasizes stochastic rounding for backward/update passes and identifies a threshold where quantization noise becomes too large relative to gradient norm.

This is one of the most relevant FP4/INT4-adjacent training papers because it treats low precision as a stability-limited training regime rather than only an inference compression trick.

## Relevance To This Project

The gradient-norm-versus-quantization-noise framing is directly useful for H6. A small LoRA calibration pass could estimate whether a module's gradients are safely above quantization noise before demoting that module to lower precision.

## Key Takeaways

- Ultra-low-precision training depends on scaling format, rounding mode, and noise-to-gradient ratio.
- Backward/update precision can matter differently from forward precision.
- Stability criteria should include quantization noise, not just NaN/Inf events.

## Evidence Gaps

- The setting is large-scale FP4 training, not small LoRA fine-tuning.
- The reported system depends on hardware and kernels outside this project's likely environment.
- It does not compare adaptive policies against hand-selected BF16/FP32 precision islands.
