# AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning

**Year:** 2026  
**Status:** Recent arXiv preprint; highly relevant but not yet established evidence.  
**Source:** https://arxiv.org/abs/2602.22268

## Key Idea

AutoQRA jointly optimizes mixed-precision bitwidths and LoRA ranks for each layer during quantized fine-tuning.

## Relevance

This is the closest source found to the user's stated idea: a predictor/optimizer over a large combinatorial space of precision and adaptation choices.

## Connection To Current Project

- Strong motivation for extending H7 from precision-only assignment to joint `bitwidth + LoRA rank + backend` assignment.
- Supports the idea that precision policy and adaptation capacity should not be optimized independently.
- Useful to cite as emerging parallel work if the paper discusses future systems directions.

## Limitations For This Project

Because this is a recent preprint, it should not be treated as settled or foundational. It is best used to position future work and avoid claiming the joint rank/bitwidth idea is unexplored.

