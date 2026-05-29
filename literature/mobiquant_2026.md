# MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Elastic LLMs

- Authors: Dongwei Wang, Jinhee Kim, Seokho Han, Denis Gudovskiy, Yohei Nakata, Tomoyuki Okuno, KhayTze Peong, Kang Eun Jeon, Jong Hwan Ko, Yiran Chen, Huanrui Yang
- Year: 2026
- Source: https://arxiv.org/abs/2602.20191
- Topic: Token-adaptive elastic mixed-bit LLM inference
- Code: No public codebase found in the current search pass.

## Key Idea

MoBiQuant proposes a Mixture-of-Bits framework for elastic LLM inference. It
observes that token-level sensitivity varies with precision because outlier
behavior changes across bitwidths, then uses two mechanisms:

- Many-in-one recursive residual quantization: reconstruct higher-precision
  weights by adding residual bit slices.
- Token-aware routing: dynamically select how many residual bit slices to use
  based on token sensitivity.

## Relevance

This is highly related to the current soft-pruning/mixed-precision idea. It is
not ordinary pruning, but it treats precision as a dynamic information budget:
easy or insensitive tokens can use fewer bits, while sensitive tokens can use
more residual bit slices.

It also addresses the storage issue directly. Instead of storing separate
checkpoints for 2-bit, 4-bit, 6-bit, etc., it stores a nested/residual
representation that can be incrementally reconstructed.

## Fit To The Proposed Direction

MoBiQuant supports all three axes of the user's formulation:

- **Prompt/token importance:** precision is selected from token sensitivity.
- **Hardware/resource elasticity:** the model can run at different runtime
  complexities.
- **Storage of different bitwidths:** recursive residual slices avoid redundant
  bit-specific model copies.

## Caveats

- Current evidence found it as an arXiv 2026 preprint, not a strongly
  peer-reviewed main-conference paper.
- No public codebase was found in the current search pass.
- It should be treated as a highly related emerging paper, not as the strongest
  reviewed or reproducible baseline.

