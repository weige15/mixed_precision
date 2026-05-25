# H8: Hardware-Aware Precision Policy Search

This branch extends H7 from module-risk prediction to hardware-aware policy selection.

Main files:

- `protocol.md`: locked H8 question, hypothesis, objective, and first experiment design.
- `analysis.md`: running synthesis.
- `code/build_h8_policy_candidates.py`: first conservative policy planner.
- `results/`: generated candidate policies and experiment outputs.

Current status: H8 has a supported first-pass result on Llama-3.1-8B. Across
500-step RTX 3090 seeds 42, 43, and 44, selective bf16 projection rescue from
QLoRA/NF4 improves final eval loss versus blanket QLoRA on every seed while
keeping about 25.3% peak-memory savings versus bf16. It is not a throughput
win: both H8 and blanket QLoRA remain about 19% slower than bf16.

First planning command:

```bash
python experiments/h8-hardware-aware-precision-search/code/build_h8_policy_candidates.py
```

Implemented policy: `run_lora_precision.py` has
`h8_qlora_nf4_rescue_projection_top4`. It starts from QLoRA/NF4, reloads the
selected projection weights from checkpoint shards into frozen bf16/fp32
`torch.nn.Linear` modules, then applies LoRA wrapping.

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

Summarize matched H8 metrics:

```bash
python experiments/h8-hardware-aware-precision-search/code/summarize_h8_llama_metrics.py
```

Close-out interpretation: H8 supports hardware-aware selective rescue as a
small quality improvement over blanket QLoRA at nearly the same memory budget.
Further Transformer inference policy search should start as H9, not as an H8
extension.
