# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

- **Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Year:** 2022 / NeurIPS 2022
- **Source:** https://arxiv.org/abs/2208.07339
- **DOI:** https://doi.org/10.48550/arXiv.2208.07339

## Summary

LLM.int8() is an inference-focused mixed-precision method for Transformer language models. Its key observation is that large LLMs develop systematic activation outlier dimensions that are hard to quantize without quality loss. The method quantizes most matrix multiplication work to INT8, but isolates outlier feature dimensions into a 16-bit path.

This makes it directly relevant to adaptive precision assignment: the paper does not use one global bit width, but derives a split from measured activation structure. Most values can be processed at INT8, while a tiny but important outlier subset receives higher precision.

## Relevance To This Project

The project's adaptive precision idea can reuse the same principle during LoRA fine-tuning: identify numerically sensitive substructures from stability signals, then promote only those paths. LLM.int8() suggests activation outlier magnitude and feature concentration are useful candidate signals.

## Key Takeaways

- Outlier features can dominate LLM quantization behavior.
- A small high-precision residual path can preserve accuracy while most compute/storage uses INT8.
- Precision assignment can be driven by observed tensor statistics rather than manual module lists.

## Evidence Gaps

- The paper targets inference, not training or LoRA fine-tuning.
- It preserves 16-bit paths for outliers but does not study dynamic promotion/demotion during an optimizer run.
- It does not connect outlier routing to training stability diagnostics such as gradient spikes or loss deltas.
