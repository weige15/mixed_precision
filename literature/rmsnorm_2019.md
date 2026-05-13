# Root Mean Square Layer Normalization

- **Authors:** Biao Zhang, Rico Sennrich
- **Year:** 2019
- **Source:** https://arxiv.org/abs/1910.07467
- **DOI:** https://doi.org/10.48550/arXiv.1910.07467

## Summary

This paper introduces RMSNorm as a simpler alternative to LayerNorm. RMSNorm removes the re-centering operation and keeps re-scaling invariance, regularizing activations by their root mean square. The authors report comparable performance to LayerNorm with lower runtime across several models.

Modern LLMs such as Qwen commonly use RMSNorm-like modules, making this paper useful for understanding what H1 targets.

## Relevance To This Project

H1's precision island targets Qwen2RMSNorm modules. RMSNorm is a reduction-heavy operation, which makes its compute dtype scientifically plausible as a stability factor even though the module is cheap compared with matrix multiplication.

## Key Takeaways

- RMSNorm stabilizes training through activation re-scaling.
- Its computation depends on reduction/statistics over hidden states.
- Because it is lightweight, fp32 execution may have small overhead if it improves stability.

## Evidence Gaps For H1

- The paper does not study BF16, FP16, FP8, or LoRA.
- It motivates why norms matter, but not what dtype their reductions require in modern LLM fine-tuning.
