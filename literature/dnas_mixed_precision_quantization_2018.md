# Mixed Precision Quantization of ConvNets via Differentiable Neural Architecture Search

**Authors:** Bichen Wu, Yanghan Wang, Peizhao Zhang, Yuandong Tian, Peter Vajda, Kurt Keutzer  
**Year:** 2018  
**Source:** https://arxiv.org/abs/1812.00090

## Key Idea

This paper casts layer-wise mixed precision quantization as a neural architecture search problem and relaxes the discrete bitwidth choices into a differentiable optimization.

## Relevance

It directly addresses the combinatorial explosion in precision assignment. Instead of enumerating all bitwidth combinations, the method uses gradient-based search over a relaxed representation.

## Connection To Current Project

- Provides a second optimization pattern besides RL: differentiable relaxation.
- Suggests H7 could eventually move from greedy ranking to soft policy optimization.
- Useful for thinking about non-additive module interactions, because architecture-search formulations can optimize a whole policy rather than only independent module scores.

## Limitations For This Project

The experiments are ConvNet-focused and mostly inference/compression oriented. For LoRA fine-tuning, a simpler greedy or knapsack optimizer is probably the right first step before differentiable search.

