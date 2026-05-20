# Low-Precision Data Formats in Large Language Models

- **Author:** Frank
- **Year:** 2026
- **Source:** https://research.frankk.site/en/llm-low-precision-formats/
- **Published:** 2026-05-18
- **Type:** Technical reference article

## Summary

This article is a broad technical reference on low-precision formats used in LLM training and inference. It compares FP32, TF32, FP16, BF16, FP8 E4M3/E5M2, FP6, FP4, INT8, and INT4 by bit layout, exponent/mantissa tradeoff, dynamic range, scaling requirements, hardware support, and typical placement inside a Transformer block.

The central framing is that low precision is not only a memory-saving technique. It is also a compute-throughput strategy: lower bit widths allow more MAC units on the same silicon, so hardware roadmaps increasingly expose faster Tensor Core or matrix-engine paths for FP8, FP6, FP4, INT8, and INT4.

The article also emphasizes that practical LLM systems mix precision by operation type rather than applying one global dtype. GEMMs are the main low-precision target, while softmax, normalization, residual accumulation, optimizer state, and some training accumulators often stay in higher precision.

## Relevance To This Project

This source is useful as a compact map of the format and hardware landscape around H6. It supports the project's hardware-realism concern: a precision policy is only valuable as an efficiency story if the selected formats match what local or target accelerators can actually execute efficiently.

It also reinforces the project's core assumption that precision sensitivity is operation-dependent. The article's Transformer-block view aligns with treating norms, softmax, residual paths, optimizer state, activations, gradients, and GEMMs as separate precision decisions rather than one layer-wise dtype switch.

## Key Takeaways

- BF16 keeps FP32-like exponent range and remains a strong training baseline when dynamic range is more important than mantissa precision.
- FP8 splits roles between E4M3, which favors precision and is often used for forward weights/activations, and E5M2, which favors range and is better matched to gradients.
- FP4 and FP6 require block scaling, such as MX or NVIDIA NVFP4, because their raw dynamic range is too small to be useful alone.
- Integer formats have no exponent and depend on external scale factors; INT4 usually needs group-wise scales from methods such as GPTQ or AWQ.
- Low precision in Transformers concentrates around GEMMs; numerically sensitive operations such as softmax, normalization, residual accumulation, and some training state usually remain higher precision.
- Hardware support is a first-order constraint: FP8 is realistic on H100/MI300-class systems, FP4/FP6 are Blackwell/MI350-era features, and TPU-style stacks remain BF16/INT8-centered.

## Implications For H6

The article strengthens the case for separating three questions in H6:

1. Which tensors or operations are numerically sensitive?
2. Which low-precision formats or quantization schemes are available in the software stack?
3. Which of those formats have real kernel and hardware support on the target accelerator?

For the current RTX 3090-class local setting, this points away from treating FP8 or FP4 training as an immediate hardware-realistic path. QLoRA, bitsandbytes INT8/4-bit paths, TorchAO/PEFT quantization, and fake-quant perturbation probes remain more plausible near-term tools. FP8/FP4 results should still inform the research story, but claims about speed or memory savings need target-hardware qualification.

## Evidence Gaps For H6

- The article is a secondary technical overview, not a controlled experiment.
- Several model-specific precision claims are based on public disclosures or author inference, so they should not be treated as direct evidence for closed models.
- It describes general Transformer precision placement but does not isolate LoRA fine-tuning, adapter gradients, or optimizer-state precision.
- It does not provide local RTX 3090 measurements, so hardware conclusions must be cross-checked against the available kernels and libraries.
