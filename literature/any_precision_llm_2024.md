# Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs

- Authors: Yeonhong Park, Jake Hyun, SangLyul Cho, Bonggeun Sim, Jae W. Lee
- Venue/year: ICML 2024 oral
- Source: https://arxiv.org/abs/2402.10517
- Code: https://github.com/SNU-ARC/any-precision-llm
- Topic: Single memory image serving multiple LLM bitwidths

## Key Idea

Any-Precision LLM overlays LLM variants quantized to different bitwidths into a
memory footprint comparable to a single high-bit model. It uses post-training
quantization plus a serving engine so multiple precisions can be deployed
without storing multiple independent checkpoints.

## Why It Is Like MoBiQuant

It shares MoBiQuant's deployment goal: one model representation should support
many precision points. The main difference is that MoBiQuant adds token-aware
routing over residual bit slices, while Any-Precision LLM focuses on overlaying
multiple bitwidth variants for serving.

## Fit To Soft Pruning

This supports the "not all information needs all bits" framing and directly
addresses storage. It is less dynamic than MoBiQuant, but it is a reviewed,
open-source baseline for multi-bit storage.

