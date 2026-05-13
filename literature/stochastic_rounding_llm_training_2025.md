# Stochastic Rounding for LLM Training: Theory and Practice

- **Authors:** Kaan Ozkara, Tao Yu, Youngsuk Park
- **Year:** 2025
- **Source:** https://arxiv.org/abs/2502.20566
- **DOI:** https://doi.org/10.48550/arXiv.2502.20566

## Summary

This paper studies stochastic rounding as a way to reduce numerical error in low-precision LLM training. It argues that mixed-precision strategies often require manual adjustment and lack sufficient theoretical grounding. The authors provide theory for stochastic rounding under Adam and report empirical results for pretraining models up to 6.7B parameters.

Their reported BF16 plus stochastic rounding strategy improves validation perplexity, throughput, and memory relative to a BF16/FP32 mixed-precision setup in their setting.

## Relevance To This Project

The paper suggests that the next level after H1 may not be "more fp32 everywhere," but better treatment of rounding error. If fp32 norms do not help, stochastic rounding or optimizer-state precision could become a more promising follow-up.

## Key Takeaways

- Rounding behavior is a meaningful part of mixed precision, not just storage dtype.
- Optimizer interaction matters; low precision can affect Adam dynamics.
- Stability and efficiency can both improve when numerical error is handled deliberately.

## Evidence Gaps For H1

- The method is not a simple PyTorch LoRA intervention.
- The paper does not isolate normalization dtype or small adapter fine-tuning.
