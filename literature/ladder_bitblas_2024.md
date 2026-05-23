# Ladder and BitBLAS: Hardware-Aware Low-Precision Tensor Transformations

**Paper:** Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation  
**Year:** 2024  
**Venue:** OSDI 2024  
**Paper page:** https://www.usenix.org/conference/osdi24/presentation/wang-lei  
**Project:** https://github.com/microsoft/BitBLAS

## Key Idea

Ladder studies hardware-aware tensor transformations for efficient low-precision deep learning. BitBLAS exposes mixed-precision matrix multiplication kernels for quantized LLM deployment, such as INT4 weights with FP16 activations.

## Relevance

This is a backend-oriented source. It is relevant because a learned precision policy only produces a real speed or memory win if the selected operations map to efficient kernels.

## Connection To Current Project

- Candidate backend for future hardware-backed matrix multiplication experiments.
- Supports the need to separate algorithmic sensitivity from hardware realizability.
- Could be useful if H7 policies move from fake-int8 output hooks to real `W_int4 A_fp16` or related kernels.

## Limitations For This Project

BitBLAS is deployment-focused. Training-time LoRA integration would need extra engineering around autograd, adapters, and optimizer state.

