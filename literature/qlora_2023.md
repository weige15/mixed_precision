# QLoRA: Efficient Finetuning of Quantized LLMs

- **Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **Year:** 2023
- **Source:** https://arxiv.org/abs/2305.14314
- **DOI:** https://doi.org/10.48550/arXiv.2305.14314

## Summary

QLoRA fine-tunes LoRA adapters while backpropagating through a frozen 4-bit quantized base model. It introduces NF4, double quantization, and paged optimizers to reduce memory enough to fine-tune very large models on limited hardware while preserving 16-bit fine-tuning performance.

Although QLoRA is partly a quantization paper, it is important for this project because it is one of the most influential examples of low-precision fine-tuning under limited GPU resources.

## Relevance To This Project

H1 intentionally starts simpler than QLoRA: BF16 LoRA with optional fp32 normalization, not 4-bit base weights. QLoRA is a nearby future direction if the project pivots from activation/norm precision to memory-constrained adapter fine-tuning.

## Key Takeaways

- Low-precision base-model storage can coexist with higher-precision adapter training.
- Memory spikes and optimizer behavior are important practical constraints.
- Fine-tuning quality may depend more on data quality and evaluation reliability than on raw scale alone.

## Evidence Gaps For H1

- QLoRA is primarily about 4-bit quantized model storage, not whether BF16 normalization compute should be fp32.
- It does not directly resolve the dtype placement question raised by the H1 dtype probe.
