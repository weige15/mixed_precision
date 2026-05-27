# PyTorch Native Architecture Optimization: torchao

**Source:** https://pytorch.org/blog/pytorch-native-architecture-optimization/  
**Organization:** PyTorch Foundation  
**Date:** 2024-09-26  

## Key Points

The post announces TorchAO as a PyTorch-native library for making models faster
and smaller with low-bit dtypes, quantization, and sparsity across inference
and training.

For Llama-family inference, the post reports:

- Llama 3 8B inference speedup from `autoquant` with int4 weight-only
  quantization and HQQ.
- Llama 3.1 8B 128K-context peak VRAM reduction with quantized KV cache.
- Composable weight and KV-cache quantization, including int4 weights plus int8
  KV cache.

The inference API examples name candidate families directly relevant to H10:

- `int4_weight_only`
- `int8_weight_only`
- `int8_dynamic_activation_int8_weight`
- `int8_dynamic_activation_int8_semi_sparse_weight`
- `float8_weight_only`
- `float8_dynamic_activation_float8_weight`

The post also warns, implicitly through `autoquant`, that not every quantized
layer is faster; layer-wise policy choice should be workload- and backend-aware.

## Relevance To H10

This helps H10 as ecosystem and candidate-space evidence, not as a direct local
result. It supports the active H10 framing: choose backend-feasible inference
policies with a measured quality, latency, memory, and workload trade-off rather
than treating bitwidth as an abstract property.

The strongest H10 implications are:

- Add TorchAO candidate actions to the H10 vocabulary even if the current vLLM
  online path fails.
- Treat `autoquant` as prior evidence that per-layer selection can matter
  because some quantized layers can be slower.
- Keep KV-cache precision as a separate action dimension from weight precision.
- Prefer artifact-backed or integration-backed validation paths when online
  quantization through a serving engine cannot load the base checkpoint.

## Limitations

The blog benchmarks use different hardware and software stacks than the local
RTX 3090/vLLM H9-H10 stack. Its reported speedups and memory reductions should
not be copied into H10 results. They can justify candidate selection and related
work only until local H9 benchmark and quality artifacts exist.

The post lists SGLang, Hugging Face Transformers, torchchat, and torchtune
integrations, but does not establish that vLLM online TorchAO quantization can
load ordinary Hugging Face safetensors. The current local TorchAO/vLLM loader
failure therefore remains valid H10 systems evidence.

## H10 Decision

Helpful, but not enough to change the immediate experimental path. The next
H10 step should remain artifact-backed PTQ ingestion through H9. The post adds
a secondary route: if vLLM online TorchAO remains blocked, validate TorchAO
policies through a supported integration such as Transformers or SGLang, then
record that as a separate backend in the H10 action table instead of forcing it
through the failed vLLM path.
