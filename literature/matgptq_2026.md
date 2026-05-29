# MatGPTQ: Accurate and Efficient Post-Training Matryoshka Quantization

- Authors: Maximilian Kleinegger, Elvir Crncevic, Dan Alistarh
- Year: 2026
- Source: https://arxiv.org/abs/2602.03537
- Code: https://github.com/IST-DASLab/MatGPTQ
- Topic: One-shot post-training Matryoshka quantization for multi-precision LLM deployment

## Key Idea

MatGPTQ creates a single sliceable quantized checkpoint that can be served at
multiple bitwidths by slicing most-significant bits. It adapts GPTQ to a
multi-precision objective with cross-bit error compensation and includes
budget-aware heterogeneous bitwidth search plus kernels for slicing and
mixed-precision execution.

## Why It Is Like MoBiQuant

Both target a single representation that supports multiple effective
precisions. MoBiQuant chooses residual bit slices dynamically by token;
MatGPTQ focuses on accurate post-training bit slicing and practical kernels.

## Fit To Soft Pruning

MatGPTQ is highly relevant to the storage side of soft-pruning-as-bitwidth:
instead of storing every precision separately, bits are nested and lower-bit
models are obtained by slicing away refinement information.

