# ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers

- **Authors:** Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He
- **Year:** 2022 / NeurIPS 2022
- **Source:** https://arxiv.org/abs/2206.01861
- **DOI:** https://doi.org/10.48550/arXiv.2206.01861

## Summary

ZeroQuant is an end-to-end Transformer quantization and inference pipeline. It combines fine-grained hardware-friendly quantization, layer-by-layer knowledge distillation, and optimized backend kernels. The reported settings include INT8 weights and activations, plus mixed INT4/INT8 configurations where fully connected modules may use INT4 while attention and activations remain INT8.

Although the paper is not about adaptive training, it demonstrates that precision should differ by module type and deployment constraint. It also highlights that algorithmic precision policy and backend support must match; otherwise conversion overhead can erase theoretical gains.

## Relevance To This Project

ZeroQuant is useful context for the INT8/INT4 part of the project. It suggests a conservative candidate hierarchy: lower precision may be more plausible for projection/feed-forward paths than for attention, normalization, or loss computation.

## Key Takeaways

- Mixed INT4/INT8 policies can be more practical than uniform ultra-low precision.
- Hardware/backend costs are part of the precision assignment objective.
- Layer-wise distillation can repair quantization damage, but this adds machinery beyond simple dtype switching.

## Evidence Gaps

- The work is primarily post-training inference quantization, not LoRA training.
- It does not provide a training-time signal policy for promoting or demoting precision.
- Its distillation-based repair path is heavier than the project's intended lightweight calibration pass.
