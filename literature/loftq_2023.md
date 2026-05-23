# LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

**Authors:** Yixiao Li et al.  
**Year:** 2023 / ICLR 2024  
**Source:** https://arxiv.org/abs/2310.08659  
**Project:** https://github.com/yxli2123/LoftQ  
**Microsoft Research blog:** https://www.microsoft.com/en-us/research/blog/loftq-reimagining-llm-fine-tuning-with-smarter-initialization/

## Key Idea

LoftQ jointly considers quantization and LoRA initialization so that LoRA adapters are initialized to compensate quantization error.

## Relevance

LoftQ is directly relevant to LoRA fine-tuning on quantized backbones. It addresses the mismatch between post-training quantization and subsequent adapter training.

## Connection To Current Project

- Suggests another route besides module precision assignment: initialize or allocate LoRA capacity to correct quantization error.
- Complements the "selective rescue from QLoRA" idea.
- Supports treating low-bit fine-tuning as a joint quantization-plus-adaptation problem.

## Limitations For This Project

LoftQ does not primarily learn a per-module precision policy from stability signals. It is a quantization-aware initialization method.

