# Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs

- Authors: Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim, Abolfazl Razi
- Year: 2026
- Source: https://arxiv.org/abs/2604.04722
- Topic: Learned per-token adaptive KV-cache precision for on-device LLMs
- Code: No public codebase found in the current search pass.

## Key Idea

This paper proposes a learned adaptive precision policy for KV-cache
quantization. A compact controller uses token-level features such as frequency,
quality score, attention variance, and entropy uncertainty to choose among
2-bit, 4-bit, 8-bit, and FP16 KV precision during decoding.

## Why It Is Like MoBiQuant

It is extremely close in spirit: both assign more bits to important/sensitive
tokens and fewer bits to low-impact tokens. MoBiQuant applies this idea to
weight residual bit slices; this paper applies it to KV-cache storage.

## Fit To Soft Pruning

This is one of the cleanest "soft pruning" analogues because it explicitly
follows variable-length coding intuition: important tokens receive more
representation budget, less important tokens receive less.

