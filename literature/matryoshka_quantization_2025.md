# Matryoshka Quantization

- Authors: Pranav Ajit Nair, Puranjay Datta, Jeff Dean, Prateek Jain, Aditya Kusupati
- Venue/year: ICML 2025
- Source: https://proceedings.mlr.press/v267/nair25a.html
- Topic: Nested multi-precision quantized model storage

## Key Idea

Matryoshka Quantization trains a single quantized model whose most significant
bits form lower-precision models. A high-precision integer representation can be
served as int8, int4, int2, or related levels by slicing bit planes rather than
storing separate checkpoints for each precision.

## Relevance

This is a central storage reference. It attacks the exact problem that storing
weights at many bitwidths can require multiple model copies. Its answer is to
make lower-precision representations nested inside higher-precision ones.

## Use For This Project

- Anchor for "store once, serve at multiple bitwidths."
- Natural bridge to bit-plane or embedding disentanglement: important
  information should be placed in high-order bits, while refinement/detail can
  live in lower-order residual bits.
- Good design inspiration for a learned representation where early bit planes
  are task-critical and later bit planes are optional.

