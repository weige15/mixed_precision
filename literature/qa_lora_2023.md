# QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

**Authors:** Xu et al.  
**Year:** 2023 / ICLR 2024  
**Source:** https://arxiv.org/abs/2309.14717  
**OpenReview PDF:** https://openreview.net/pdf?id=WvFoJccpo8  
**Project:** https://github.com/yuhuixu1993/qa-lora

## Key Idea

QA-LoRA proposes quantization-aware low-rank adaptation, balancing quantization and adaptation degrees of freedom so fine-tuned models can be deployed efficiently.

## Relevance

This is directly adjacent to QLoRA and H7: it connects low-rank adaptation with quantization-aware deployment constraints.

## Connection To Current Project

- Supports thinking of precision and LoRA capacity as coupled decisions.
- Relevant to a future optimizer that assigns both bitwidth and rank per layer or module.
- Helps motivate moving from a pure module-risk predictor to a joint resource allocation model.

## Limitations For This Project

QA-LoRA is not primarily a calibration-driven per-module precision predictor. It should be treated as adjacent QPEFT literature rather than a direct predecessor.

