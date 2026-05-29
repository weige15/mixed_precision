# Prompt-Adaptive Quantization: Adaptive Per-Prompt Routing for Efficient LLM Inference

- Authors: Gabriel Jimenez, Vivann Khanna, Rishi Sastri, Raine Ma, Soham Chatterjee, Kevin Zhu, Sunishchal Dev
- Venue/year: AAAI 2026 AIR-FM Workshop
- Source: https://openreview.net/forum?id=YWn5CbBSKj
- Topic: Per-prompt routing among pre-quantized LLM variants

## Key Idea

Prompt-Adaptive Quantization trains a lightweight router to choose the smallest
adequate model precision for each prompt, using pre-quantized 2-, 4-, 8-, and
16-bit model variants. The underlying LLM does not need to be retrained.

## Relevance

This paper is directly aligned with "input prompt determines bitwidth," but its
storage trade-off is different from QAQ or Matryoshka Quantization: it routes
among separately pre-quantized variants. That makes it easy to implement but
less satisfying if the goal is one compact representation.

## Use For This Project

- Use as a baseline design for prompt-conditioned precision.
- Treat multiple pre-quantized checkpoints as the simple-but-expensive storage
  solution.
- Contrast with the proposed disentangled/nested representation, which aims to
  avoid keeping several model variants.

