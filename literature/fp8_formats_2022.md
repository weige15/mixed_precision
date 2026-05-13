# FP8 Formats for Deep Learning

- **Authors:** Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, Hao Wu
- **Year:** 2022
- **Source:** https://arxiv.org/abs/2209.05433
- **DOI:** https://doi.org/10.48550/arXiv.2209.05433

## Summary

This paper proposes FP8 formats for deep learning, especially E4M3 and E5M2. It frames FP8 as a step beyond FP16/BF16 for accelerating training and inference. The authors report that FP8 can match 16-bit training quality across CNNs, RNNs, and Transformer-based models, including language models up to 175B parameters, without changing hyperparameters from 16-bit baselines.

The paper is important because it makes mixed precision more granular: different FP8 formats have different exponent/mantissa tradeoffs and may be appropriate for different tensor roles.

## Relevance To This Project

Although H1 is BF16-versus-fp32-norm rather than FP8, FP8 work sharpens the same question: which tensors and operations tolerate lower precision, and which need high precision? The paper supports the idea that format choice should be operation-aware.

## Key Takeaways

- FP8 is not one format; E4M3 and E5M2 encode different range/precision tradeoffs.
- Transformer training can tolerate aggressive low precision in some settings.
- The success of low precision depends on careful assignment of formats across computation roles.

## Evidence Gaps For H1

- It studies large-scale training and broad FP8 policies, not a cheap LoRA fine-tuning setting.
- It does not isolate RMSNorm/LayerNorm dtype under PyTorch autocast.
