# H10: PEFT Precision-Risk Prediction

H10 studies backend-aware precision assignment for LoRA/QLoRA fine-tuning. It
uses existing H6/H7 perturbation artifacts to select high-risk projection modules
that should be rescued from a QLoRA/NF4 baseline.

Build conservative seed-aggregated labels:

```bash
python experiments/h10-peft-precision-risk/code/build_seed_aggregated_dataset.py
```

Evaluate equal-budget rescue selectors and emit policy candidates:

```bash
python experiments/h10-peft-precision-risk/code/evaluate_rescue_selectors.py \
  --model-name meta-llama/Llama-3.1-8B \
  --top-k 4
```

Run the same selector screen for Qwen2.5-7B:

```bash
python experiments/h10-peft-precision-risk/code/evaluate_rescue_selectors.py \
  --model-name Qwen/Qwen2.5-7B \
  --top-k 4 \
  --output experiments/h10-peft-precision-risk/results/rescue_selector_evaluation_qwen25_7b.json \
  --policies-output experiments/h10-peft-precision-risk/results/h10_rescue_policy_candidates_qwen25_7b.json
```

The generated policies are planning artifacts. Only GPU training runs against
matched bf16 and QLoRA baselines can support a final resource-quality claim.
