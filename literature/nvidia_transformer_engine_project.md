# NVIDIA Transformer Engine

**Project / docs:** https://nvidia.github.io/TransformerEngine/  
**FP8 and FP4 primer:** https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html

## Key Idea

NVIDIA Transformer Engine provides production-oriented support for low-precision Transformer training and inference, especially FP8 and FP4 recipes on supported NVIDIA GPUs.

## Relevance

This is one of the most important hardware-backed mixed precision references because it defines what modern NVIDIA GPUs can accelerate directly.

## Connection To Current Project

- If H7 moves to FP8/FP4 policies, Transformer Engine is a likely backend on H100/H200-class systems.
- It reinforces that hardware generation matters: an RTX 3090 result cannot answer an H100 FP8 systems question.
- Provides a route for real hardware-backed precision assignment if compute access changes.

## Limitations For This Project

The current lab RTX 3090 does not provide the same FP8/FP4 training path as Hopper-class GPUs. Transformer Engine is therefore a future backend rather than an immediate local validation path.

