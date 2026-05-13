# Convergence-Aware Operator-Wise Mixed-Precision Training

- **Authors:** Wenhao Dai, Ziyi Jia, Yuesi Bai, and Qingxiao Sun
- **Year:** 2025
- **Source:** https://link.springer.com/article/10.1007/s42514-024-00208-9
- **DOI:** https://doi.org/10.1007/s42514-024-00208-9

## Summary

This paper studies operator-wise mixed-precision training under emerging hardware with multiple low-precision formats. The key framing is convergence awareness: low precision should be assigned in a way that preserves training convergence, not only local operator speed.

This is important for adaptive precision assignment because it shifts the target from static layer categories to training outcomes. A precision decision should be evaluated by its effect on convergence, quality, memory, and bandwidth.

## Relevance To This Project

H6 can be viewed as an LLM LoRA version of convergence-aware operator-wise precision. The project should collect operator-level signals, but final policy decisions must be judged by validation loss and instability counts after a fixed training budget.

## Key Takeaways

- Operator granularity is a natural level for precision assignment.
- Precision decisions should consider convergence risk, not just immediate performance.
- Emerging hardware format diversity makes manual dtype rules brittle.

## Evidence Gaps

- The paper is not specific to LLM LoRA fine-tuning.
- It does not settle which cheap stability signals are predictive for Transformer modules.
- The project still needs a minimal implementation that works in ordinary PyTorch/Transformers.
