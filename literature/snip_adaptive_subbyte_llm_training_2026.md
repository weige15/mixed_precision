# SNIP: An Adaptive Mixed Precision Framework for Subbyte Large Language Model Training

- **Authors:** Yunjie Pan, Yongyi Yang, Hanmei Yang, Scott Mahlke
- **Year:** 2026
- **Source:** https://arxiv.org/abs/2602.01410
- **DOI:** https://doi.org/10.48550/arXiv.2602.01410

## Summary

SNIP is a recent adaptive mixed-precision framework for subbyte LLM pretraining. It periodically collects statistics on activations, gradients, and optimizer states, estimates precision-loss impact through forward loss divergence and backward weight divergence, then solves an integer linear programming problem to choose layer-wise precision under efficiency targets.

This is the closest match to the project's research question. It explicitly connects adaptive precision, LLM training, stability/convergence, and subbyte precision.

## Relevance To This Project

H6 can be framed as a lightweight LoRA-fine-tuning analogue of SNIP. Instead of full pretraining and ILP at large scale, the project can test whether cheap calibration signals predict which LoRA/Transformer operations need BF16 or FP32 and which paths tolerate INT8/INT4-style perturbations.

## Key Takeaways

- Adaptive LLM precision can use activation, gradient, and optimizer-state statistics.
- Forward loss divergence and backward update divergence are strong conceptual signals.
- Policy decisions should be periodically recalibrated, but final comparisons require a frozen policy.

## Evidence Gaps

- SNIP targets LLM pretraining rather than adapter fine-tuning.
- Its infrastructure and subbyte kernels may be unavailable on the local RTX 4050 setup.
- It does not isolate normalization, logits/loss, or LoRA adapter precision under BF16 baselines.
