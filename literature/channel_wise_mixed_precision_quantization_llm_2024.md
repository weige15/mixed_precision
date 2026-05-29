# Channel-Wise Mixed-Precision Quantization for Large Language Models

- Authors: Zihan Chen, Bike Xie, Jundong Li, Cong Shen
- Year: 2024, revised 2025
- Source: https://arxiv.org/abs/2410.13056
- Topic: Activation-distribution-based channel-wise mixed precision

## Key Idea

Channel-Wise Mixed-Precision Quantization allocates weight precision by channel
using activation distributions. It targets both integer and fractional average
bitwidth constraints and uses outlier extraction to preserve critical
information.

## Relevance

This paper supports a middle granularity between layer-wise assignment and
per-weight assignment. Channel-wise precision is more expressive than uniform
group quantization but still more structured than arbitrary per-weight bitwidths.

## Use For This Project

- Cite as evidence for activation-driven importance.
- Use channel-wise structure as a practical granularity for disentangled
  important/less-important subspaces.
- Relevant to fractional average bit budgets where a layer cannot simply be all
  2-bit, 4-bit, or 8-bit.

