# Mixed Precision Training

- **Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu
- **Year:** 2017 / ICLR 2018
- **Source:** https://arxiv.org/abs/1710.03740
- **DOI:** https://doi.org/10.48550/arXiv.1710.03740

## Summary

This is the canonical modern mixed-precision training paper. It proposes storing weights, activations, and gradients in FP16 while preserving training quality through two stabilizers: maintaining FP32 master weights for accumulation and using loss scaling to prevent small FP16 gradients from underflowing.

The paper reports broad success across CNNs, RNNs, GANs, and large models, with memory consumption reduced by nearly 2x. Its central lesson is that mixed precision is not simply "use low precision everywhere"; it is a policy about which tensors can be low precision and which accumulations or updates must retain more precision.

## Relevance To This Project

H1 is a small version of the same principle: identify precision-sensitive operations rather than applying one blanket dtype. Our fp32-norm treatment echoes the paper's idea of keeping specific numerically important operations or states in higher precision.

## Key Takeaways

- FP16 training needed FP32 master weights and loss scaling because FP16 has limited dynamic range.
- The scientific unit is not just dtype, but dtype placement across the training computation.
- Stability interventions should be evaluated against memory and speed overhead.

## Evidence Gaps For H1

- The paper is pre-LLM and does not isolate RMSNorm/LayerNorm dtype in Transformer LoRA fine-tuning.
- It focuses on FP16 rather than BF16, and BF16 changes the stability story because it has FP32-like exponent range.
