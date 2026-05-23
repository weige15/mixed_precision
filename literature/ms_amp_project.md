# MS-AMP: Microsoft Automatic Mixed Precision

**Project:** https://github.com/Azure/MS-AMP  
**Docs:** https://azure.github.io/MS-AMP/docs/introduction/  
**Related paper:** FP8-LM, https://arxiv.org/abs/2310.18313

## Key Idea

MS-AMP is Microsoft's automatic mixed precision library and is connected to FP8-LM experiments on large-scale LLM training.

## Relevance

MS-AMP is useful as a concrete example of FP8 mixed-precision training infrastructure rather than just a quantization algorithm.

## Connection To Current Project

- Shows that hardware-backed low precision for training is possible when the software stack and hardware match.
- Useful comparison point if the project later targets H100/A100-style distributed training rather than single-GPU RTX 3090 LoRA.
- Supports the idea that precision assignment should be backend-aware.

## Limitations For This Project

MS-AMP is not designed around small module-wise LoRA rescue policies. It is more relevant to large-scale FP8 mixed training infrastructure.

