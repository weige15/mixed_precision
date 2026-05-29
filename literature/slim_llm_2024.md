# SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models

- Authors: Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Qinshuo Liu, Xianglong Liu, Luca Benini, Michele Magno, Shiming Zhang, Xiaojuan Qi
- Year: 2024, revised 2025
- Source: https://arxiv.org/abs/2405.14917
- Topic: Salience-driven group-wise mixed-precision PTQ for LLMs

## Key Idea

SliM-LLM assigns bitwidths to weight groups according to salience. It combines
salience-determined bit allocation with salience-weighted quantizer calibration,
then uses structured partitioning to preserve hardware friendliness.

## Relevance

This paper is directly aligned with the user's "important information gets more
bits, less important information gets fewer bits" framing. It also makes the
practical point that element-wise high-precision exceptions can be accurate but
hard to realize efficiently, so salience should be grouped in a compact layout.

## Use For This Project

- Strong reference for importance/salience-based bit allocation.
- Useful contrast for disentanglement: if importance is clustered or can be made
  clustered, mixed precision becomes easier to pack and execute.
- Supports targeting group/channel/block granularity before considering truly
  per-weight bit allocation.

