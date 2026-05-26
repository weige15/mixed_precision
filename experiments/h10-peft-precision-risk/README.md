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

## Next Step Commands

Current next run: 500-step seed-42 validation for the perturbation-guided H10
upper-bound policy. The 100-step screen showed `h10_oracle_perturbation_top4`
improves over blanket QLoRA and the previous H8 top-3 rescue, while
`h10_activation_outlier_top4` was worse than blanket QLoRA.

```bash
GPU_ID=0 \
HARDWARE_LABEL=rtx3090-lab \
SETUP_ONLY=0 \
SEEDS=42 \
MAX_STEPS=500 \
EVAL_MAX_BATCHES=100 \
H10_POLICIES="h10_oracle_perturbation_top4" \
bash experiments/h10-peft-precision-risk/code/run_llama31_8b_h10_rescue_controls.sh
```

If the 500-step seed-42 run remains better than blanket QLoRA and stays inside
the 1% bf16 quality gate, replicate the same policy on seeds 43 and 44:

```bash
GPU_ID=0 \
HARDWARE_LABEL=rtx3090-lab \
SETUP_ONLY=0 \
SEEDS="43 44" \
MAX_STEPS=500 \
EVAL_MAX_BATCHES=100 \
H10_POLICIES="h10_oracle_perturbation_top4" \
bash experiments/h10-peft-precision-risk/code/run_llama31_8b_h10_rescue_controls.sh
```
