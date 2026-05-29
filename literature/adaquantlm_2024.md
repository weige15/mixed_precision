# AdaQuantLM: LLM Quantization with Adaptive Bit-Widths

- Authors: Shuangyi Chen, Ashish J. Khisti
- Venue/year: NeurIPS 2024 Compression Workshop
- Source: https://openreview.net/forum?id=fnHOTCcq2Y
- Topic: Adaptive bit-width LLM quantization through additive codewords

## Key Idea

AdaQuantLM uses additive codewords so a quantized LLM can move between bitwidths
by adding or removing codewords. It jointly quantizes and fine-tunes across
multiple bitwidths, aiming to avoid separate fine-tuning and benchmarking for
each target precision.

## Relevance

This is another storage-efficient adaptive-precision idea. Unlike per-prompt
routing to multiple separate checkpoints, it points toward a single compressed
representation that can be expanded or reduced.

## Use For This Project

- Cite for "adaptive bitwidth without keeping full-precision weights or separate
  model variants."
- Connect additive codewords to the proposed disentanglement direction:
  base codewords store essential information, extra codewords store refinements.
- Useful bridge between Matryoshka bit planes and vector-quantized embeddings.

