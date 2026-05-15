# Hardware-Realistic Low Precision for H6

## Why This Matters

H6 currently has strong fake-int8 evidence: calibration signals identify MLP projection outputs that tolerate quantization noise and high-risk modules that do not. The missing claim is resource benefit. Fake quantization does not use packed low-bit storage or hardware low-precision kernels, so it cannot establish speed or memory savings.

On the available RTX 3090-class Ampere hardware, the most realistic near-term resource route is not FP8/FP4 Transformer Engine training. Transformer Engine targets FP8 on Hopper, Ada, and Blackwell, while Ampere support is primarily for FP16/BF16 optimizations. The near-term hardware-realistic candidates are quantized frozen base weights plus LoRA adapters, quantized optimizers, and possibly TorchAO/PEFT int8 weight-only quantization.

## Strongest Implementation Candidates

### bitsandbytes / QLoRA

- Sources: https://github.com/bitsandbytes-foundation/bitsandbytes, https://github.com/artidoro/qlora, https://arxiv.org/abs/2305.14314
- Relevance: Most mature path for real fine-tuning memory savings. QLoRA stores the frozen pretrained model in 4-bit NF4, backpropagates through it into LoRA adapters, and uses double quantization plus paged optimizers.
- Fit to H6: Good for a resource branch. It does not directly implement H6's selected output fake-int8 modules, but it provides a hardware-realistic baseline: bf16 LoRA versus quantized-base LoRA using the same calibration-selected safe/unsafe module analysis.
- Local next test: add a `qlora_4bit_nf4` policy to the runner using `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` and `prepare_model_for_kbit_training`.

### TorchAO + PEFT

- Sources: https://github.com/pytorch/ao, https://huggingface.co/docs/peft/developer_guides/quantization, https://huggingface.co/docs/transformers/quantization/torchao
- Relevance: PEFT explicitly supports TorchAO int8 quantized models with LoRA, and TorchAO is a PyTorch-native route for quantization, QAT, quantized optimizers, and training-to-serving workflows.
- Fit to H6: Good conceptual match because H6 already derives per-module policies. TorchAO may eventually let the project express those policies with real tensor subclasses and quantization configs rather than Python hooks.
- Caveat: the current environment has shown a TorchAO/PyTorch extension compatibility warning, so this path may need version cleanup before it is a reliable experiment path.

### TorchAO QAT / Unsloth / Axolotl integrations

- Sources: https://pytorch.org/blog/quantization-aware-training/, https://pytorch.org/blog/quantization-aware-training-in-torchao-ii/
- Relevance: QAT trains with fake quantization that is intended to match an actual post-training quantized deployment format. Recent TorchAO work connects QAT to CUDA-oriented inference kernels and LoRA fine-tuning workflows in Unsloth and Axolotl.
- Fit to H6: Strong research fit. H6 can use calibration to decide which modules to quantize or skip, then use QAT as the training mechanism before converting to a deployable low-bit model.
- Caveat: this is heavier than a QLoRA baseline and may require changing training frameworks.

### LR-QAT

- Sources: https://arxiv.org/abs/2406.06385, https://github.com/Qualcomm-AI-research/lr-qat
- Relevance: Low-rank quantization-aware training for LLMs uses low-rank auxiliary weights, fixed-point or double-packed downcasting, and checkpointing. The paper reports training 7B models on a 24GB consumer GPU.
- Fit to H6: Very relevant methodologically: H6's calibration-selected precision policy could be framed as a lighter module-sensitivity front end for low-rank QAT.
- Caveat: this is a separate algorithm and codebase, not a small patch to the current runner.

### EfficientQAT

- Sources: https://arxiv.org/abs/2407.11062, https://github.com/OpenGVLab/EfficientQAT
- Relevance: Efficient QAT implementation for LLMs with model transfer to deployment formats such as GPTQ v2 and BitBLAS, plus low-bit integer/float format comparisons.
- Fit to H6: Useful for understanding how calibration and QAT outputs become deployable models. It is less directly compatible with the current PEFT LoRA runner.

### IntLoRA

- Sources: https://arxiv.org/abs/2410.21759, https://github.com/csguoh/IntLoRA
- Relevance: Fine-tunes directly on quantized INT4 models and keeps both pretrained and low-rank weights in integer representations for storage and inference merging.
- Fit to H6: The idea is highly relevant, but the current official work targets diffusion models rather than LLM LoRA.

### Transformer Engine

- Source: https://github.com/NVIDIA/TransformerEngine
- Relevance: Mature FP8/FP4 training infrastructure for transformer models on newer NVIDIA GPUs.
- Fit to H6: Not the best local route for RTX 3090. FP8 is emphasized for Hopper/Ada/Blackwell; Ampere support is mainly for FP16/BF16 optimizations.

## Recommended H6 Resource Branch

1. First implement and run a `qlora_4bit_nf4` baseline/treatment path in the current runner.
2. Compare bf16 LoRA versus QLoRA on the same model, dataset split, steps, effective batch, and seeds.
3. Record final eval loss, memory, throughput, and stability. This directly tests whether a real low-bit path gives resource savings on the available 24GB GPU.
4. Then test whether H6 calibration signals still predict which modules are fragile under the QLoRA path. This keeps the H6 novelty: calibration-guided precision policy, not merely "use QLoRA."
5. Treat TorchAO/PEFT int8 and QAT as the second branch if QLoRA gives a solid resource baseline but is too coarse-grained for H6's per-module policy story.

## Current Judgment

The highest-probability next experiment is bitsandbytes QLoRA because it is mature, local-GPU friendly, and directly tied to LoRA fine-tuning. The most scientifically aligned longer path is TorchAO QAT or TorchAO+PEFT int8, but it is more likely to require dependency and framework work before producing clean measurements.
