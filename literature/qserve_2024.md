# QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving

**Authors:** MIT HAN Lab / collaborators  
**Year:** 2024 / MLSys 2025  
**Source:** https://arxiv.org/abs/2405.04532  
**Project:** https://github.com/mit-han-lab/qserve

## Key Idea

QServe co-designs quantization and serving kernels around W4A8KV4: 4-bit weights, 8-bit activations, and 4-bit KV cache.

## Relevance

QServe is important because it treats quantization as an algorithm-plus-system problem. The precision policy is only useful if the runtime can exploit it.

## Connection To Current Project

- Reinforces that H7 should predict risk conditioned on backend/format, not just abstract bitwidth.
- Suggests any hardware-backed selective precision branch should benchmark with a system that actually supports the target quantized operations.
- Useful source for explaining why fake quantization can validate sensitivity but cannot validate throughput or memory.

## Limitations For This Project

QServe is an inference serving system. It is not a LoRA fine-tuning backend.

