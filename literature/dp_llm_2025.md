# DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment

- Authors: Sangwoo Kwon, Seong Hoon Seo, Jae W. Lee, Yeonhong Park
- Venue/year: NeurIPS 2025
- Source: https://arxiv.org/abs/2508.06041
- OpenReview/PDF: https://openreview.net/pdf?id=ppKDXf55lY
- Code: https://github.com/SNU-ARC/DP-LLM
- Topic: Runtime dynamic layer-wise precision assignment for LLMs

## Key Idea

DP-LLM augments each LLM linear layer with a lightweight precision selector. At
runtime, the selector estimates the layer's quantization error from input values
and chooses the bitwidth for that layer. This turns precision into a dynamic
per-decoding-step decision rather than a static post-training assignment.

## Why It Is Like MoBiQuant

MoBiQuant routes tokens to residual bit slices; DP-LLM routes layers to
bitwidths based on runtime inputs. Both treat precision as an adaptive resource
whose allocation depends on the current inference state.

## Fit To Soft Pruning

DP-LLM is a strong mixed-precision analogue of soft pruning. A layer is not
hard-kept or removed; instead, it is assigned a soft resource level at runtime:
low bits for tolerant states and higher bits for sensitive states.

