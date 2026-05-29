# NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs

- Authors: Haeun Lee, Omin Kwon, Yeonhong Park, Jae W. Lee
- Venue/year: NeurIPS 2025
- Source: https://arxiv.org/abs/2506.02024
- OpenReview: https://openreview.net/forum?id=WDAKFpWftI
- Code: https://github.com/SNU-ARC/NestedFP
- Topic: Memory-efficient FP16/FP8 dual-precision LLM serving

## Key Idea

NestedFP overlays FP8 parameters onto FP16 parameters so both precision modes
share the same FP16 memory footprint. A specialized GEMM kernel supports
efficient execution in both modes, enabling SLO-aware switching between FP16 and
FP8 without storing two model copies.

## Why It Is Like MoBiQuant

NestedFP is less fine-grained than MoBiQuant, but it addresses the same storage
pain point: dynamic precision normally requires multiple model copies, and a
nested representation can remove that overhead.

## Fit To Soft Pruning

NestedFP is a systems baseline for "store high precision once, expose lower
precision when needed." It supports hardware-aware precision routing under load.

