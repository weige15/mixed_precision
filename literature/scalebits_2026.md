# ScaleBITS: Scalable Bitwidth Search for Hardware-Aligned Mixed-Precision LLMs

- Authors: Xinlin Li, Timothy Chou, Josh Fromm, Zichang Liu, Yunjie Pan, Christina Fragouli
- Year: 2026
- Source: https://arxiv.org/abs/2602.17698
- Topic: Hardware-aligned mixed-precision LLM weight quantization

## Key Idea

ScaleBITS treats LLM mixed-precision quantization as fine-grained bitwidth
allocation under a memory budget, but constrains the layout so the resulting
policy remains hardware efficient. It uses sensitivity analysis, block-wise
weight partitioning, bi-directional channel reordering, and a scalable
approximation to greedy constrained optimization.

## Relevance

This is one of the closest papers for the storage/layout difficulty. It argues
that irregular fine-grained mixed precision can improve accuracy but often adds
runtime overhead, so the allocation must be aligned to blocks and channels that
hardware can actually consume.

## Use For This Project

- Cite as a recent LLM-specific answer to "different weights need different
  bits, but naive irregular layouts are hard to serve."
- Use its hardware-aligned block partitioning as evidence that the research
  question should optimize quality risk and memory layout together.
- Compare against a proposed disentanglement approach: instead of arbitrary
  per-weight bitwidths, split weights or channels into structured groups whose
  bit planes can be packed regularly.

