# TorchAO: PyTorch-Native Training-to-Serving Model Optimization

**Paper:** https://arxiv.org/abs/2507.16099  
**Project:** https://github.com/pytorch/ao  
**Docs:** https://docs.pytorch.org/ao/stable/  
**QAT docs:** https://docs.pytorch.org/ao/stable/workflows/qat.html

## Key Idea

TorchAO is a PyTorch-native optimization stack for quantization, sparsity, QAT, PTQ, FP8, INT4, INT8, and MX formats. It uses tensor abstractions and quantization APIs designed to connect training-time and serving-time optimization.

## Relevance

TorchAO is a trustworthy systems source because it is part of the PyTorch ecosystem and is actively documenting QAT and low-bit workflows.

## Connection To Current Project

- Potential backend for converting H7 policies into real quantized module configs.
- `FqnToConfig`-style configuration is conceptually aligned with module-wise precision assignment.
- Useful for future QAT or fake-quant-to-real-quant experiments.

## Limitations For This Project

Backend support varies by format, device, and model pattern. A policy that is expressible in TorchAO still needs matched hardware benchmarking before claiming speed or memory gains.

