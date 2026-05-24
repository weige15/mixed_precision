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

Next implementation step: verify backend feasibility for selective rescue from
QLoRA/NF4 before launching long H8 training runs.

Backend feasibility commands:

```bash
python experiments/h8-hardware-aware-precision-search/code/inspect_h8_backend_feasibility.py \
  --model-name meta-llama/Llama-3.1-8B \
  --policy-name h8_rescue_norm_logits

python experiments/h8-hardware-aware-precision-search/code/inspect_h8_backend_feasibility.py \
  --model-name Qwen/Qwen2.5-7B \
  --policy-name h8_rescue_norm_logits
```
