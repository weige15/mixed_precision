# Adaptive Precision Training (AdaPT): A Dynamic Fixed Point Quantized Training Approach for DNNs

- **Authors:** Lorenz Kummer, Kevin Sidak, Tabea Reichmann, Wilfried Gansterer
- **Year:** 2021
- **Source:** https://arxiv.org/abs/2107.13490
- **DOI:** https://doi.org/10.48550/arXiv.2107.13490

## Summary

AdaPT studies dynamic fixed-point quantized training. It addresses the limitation of using a single global word length by assigning precision at a finer granularity during training. The motivation is close to this project: full precision everywhere wastes resources, but uniform low precision can damage convergence.

The paper is not focused on LLMs or modern BF16/FP8 hardware, but it is directly relevant as an explicit adaptive precision training formulation.

## Relevance To This Project

AdaPT supports the idea that precision can be changed during training rather than chosen once before training. It motivates tracking whether per-layer precision choices remain valid as optimization progresses.

## Key Takeaways

- Training-time precision assignment is distinct from post-training inference quantization.
- Finer-grained precision choices can reduce resource use while preserving convergence.
- Dynamic policies need stability safeguards because training distributions change over time.

## Evidence Gaps

- The method uses fixed-point quantization and does not target LLM LoRA fine-tuning.
- It does not provide a ready-made signal set for Transformer operations such as RMSNorm, attention softmax, logits, or adapter gradients.
- Its results may not transfer to GPU Tensor Core BF16/FP16/FP8 execution.
