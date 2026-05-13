# To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability

- **Authors:** Joonhyung Lee, Jeongin Bae, Byeongwook Kim, Se Jung Kwon, Dongsoo Lee
- **Year:** 2024 / revised 2025
- **Source:** https://arxiv.org/abs/2405.18710
- **DOI:** https://doi.org/10.48550/arXiv.2405.18710

## Summary

This paper directly studies reduced-precision effects on LLM training stability. The authors argue that reduced-precision schemes must preserve not only final quality but also robustness across seeds, learning rates, and datasets to be economically useful. They express concern that FP8 methods may not yet be robust enough to simply replace higher-precision training.

The work is particularly relevant because it treats instability as a first-class metric rather than looking only at final loss or throughput. It also studies progressive bit reduction and links representation power to training stability.

## Relevance To This Project

This paper strongly supports H1's measurement design. Our pilot tracks loss spikes, NaN/Inf counts, gradient norms, and throughput, rather than only final validation loss. It also justifies small stress tests as meaningful if they reveal precision-induced instability.

## Key Takeaways

- Reduced precision must be judged by stability and hyperparameter sensitivity, not only final quality.
- BF16 is common in LLM training, but lower precision can increase robustness risks.
- Seed and learning-rate sensitivity are important evidence dimensions.

## Evidence Gaps For H1

- The paper focuses on pretraining-like LLM training, not LoRA adaptation.
- It does not answer whether a selective fp32 norm island improves stability under BF16 fine-tuning.
