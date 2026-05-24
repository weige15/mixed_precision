# H8: Hardware-Aware Precision Policy Search

This branch extends H7 from module-risk prediction to hardware-aware policy selection.

Main files:

- `protocol.md`: locked H8 question, hypothesis, objective, and first experiment design.
- `analysis.md`: running synthesis.
- `code/build_h8_policy_candidates.py`: first conservative policy planner.
- `results/`: generated candidate policies and experiment outputs.

Current status: BF16 and blanket QLoRA/NF4 baselines are validated for
Llama-3.1-8B seed 42 on the RTX 3090 lab machine. The policy candidate file
now aligns Qwen2.5-7B and Llama-3.1-8B from the combined H7 dataset.

First planning command:

```bash
python experiments/h8-hardware-aware-precision-search/code/build_h8_policy_candidates.py
```

Current implementation step: `run_lora_precision.py` now has a prototype
`h8_qlora_nf4_rescue_projection_top4` policy. It starts from QLoRA/NF4,
reloads the selected projection weights from checkpoint shards into frozen
bf16/fp32 `torch.nn.Linear` modules, then applies LoRA wrapping.

Backend feasibility commands:

```bash
python experiments/h8-hardware-aware-precision-search/code/inspect_h8_backend_feasibility.py \
  --model-name meta-llama/Llama-3.1-8B \
  --policy-name h8_rescue_norm_logits

python experiments/h8-hardware-aware-precision-search/code/inspect_h8_backend_feasibility.py \
  --model-name Qwen/Qwen2.5-7B \
  --policy-name h8_rescue_norm_logits
```

Selective-rescue setup smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/h1-selective-fp32-norms/code/run_lora_precision.py \
  --model-name meta-llama/Llama-3.1-8B \
  --precision-policy h8_qlora_nf4_rescue_projection_top4 \
  --h8-policy-name h8_rescue_projection_top4 \
  --h8-rescue-precision bf16 \
  --setup-only \
  --output-dir experiments/h8-hardware-aware-precision-search/results/llama31_8b_h8_setup_seed42_rtx3090-lab
```

Short selective-rescue run:

```bash
GPU_ID=0 HARDWARE_LABEL=rtx3090-lab RUN_BF16=0 RUN_QLORA=0 RUN_SELECTIVE_RESCUE=1 \
MAX_STEPS=100 EVAL_MAX_BATCHES=25 \
bash experiments/h8-hardware-aware-precision-search/code/run_llama31_8b_h8_metrics.sh
```
