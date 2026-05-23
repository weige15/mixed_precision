# SqueezeLLM: Dense-and-Sparse Quantization

**Authors:** SqueezeAILab / UC Berkeley authors  
**Year:** 2023 / ICML 2024  
**Source:** https://arxiv.org/abs/2306.07629  
**Project:** https://github.com/SqueezeAILab/SqueezeLLM

## Key Idea

SqueezeLLM combines dense quantization with sparse handling of sensitive/outlier components to support efficient LLM serving.

## Relevance

The method reinforces the same design pattern as OWQ and SpQR: retain a structured exception path for difficult values rather than forcing a uniform bitwidth.

## Connection To Current Project

- Provides another reference for exception-aware low-bit LLM deployment.
- Supports adding sparse or structured exception cost terms to a future precision optimizer.
- Strengthens the argument that module-level precision assignment is only one granularity; column/sparse exception granularity may be more hardware-effective.

## Limitations For This Project

The focus is serving, not LoRA fine-tuning. It is useful as a hardware-backed compression reference rather than a direct training recipe.

