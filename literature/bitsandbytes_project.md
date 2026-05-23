# bitsandbytes

**Project:** https://github.com/bitsandbytes-foundation/bitsandbytes  
**Docs:** https://huggingface.co/docs/bitsandbytes/main/en/index

## Key Idea

bitsandbytes provides practical PyTorch quantization primitives for LLMs, including `Linear8bitLt`, `Linear4bit`, LLM.int8(), QLoRA-style 4-bit paths, and 8-bit optimizers.

## Relevance

This is the most directly relevant practical backend because the current project already used bitsandbytes-style 8-bit and QLoRA baselines.

## Connection To Current Project

- Confirms QLoRA/NF4 as a hardware-backed memory-capacity baseline.
- Gives a practical route for low-bit fine-tuning without writing kernels.
- Local experiments already show the trade-off: 7B QLoRA saves memory but is slower on RTX 3090.

## Limitations For This Project

bitsandbytes is not a general selective per-module precision optimizer. It supports useful low-bit building blocks, but module-wise rescue/demotion policies may require custom wrapping and careful dispatch benchmarking.

