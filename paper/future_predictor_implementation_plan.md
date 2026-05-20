# Future Implementation Plan: Precision Assignment Predictor

Draft date: 2026-05-20

This plan is intentionally separate from the current paper draft. The current evidence supports calibration-guided sensitivity ranking. The next implementation should turn that ranking into an explicit predictor and optimizer.

## Goal

Build a small research artifact that predicts module precision risk from calibration features and uses that prediction to choose a frozen precision policy.

## Data Table

Create one row per `(model, seed, module, candidate_format)`:

```text
model_name
model_size
seed
module_name
layer_idx
normalized_depth
module_role
candidate_format
activation_outlier_score
input_outlier_score
output_outlier_score
int8_rel_mse
perturbation_delta
abs_perturbation_delta
selected_by_policy
training_eval_delta_if_available
safe_label
```

Suggested labels:

```text
safe_label = abs_perturbation_delta <= 0.005
```

or, when training data exists:

```text
safe_label = training_eval_delta_relative <= 0.01
```

Do not mix these without marking the label source.

## First Predictor

Start with interpretable models:

1. Logistic regression for `safe_label`.
2. Ridge regression for `abs_perturbation_delta`.
3. Random forest or gradient boosting only after the table is stable.

Initial features:

```text
normalized_depth
one-hot module_role
log activation_outlier_score
log int8_rel_mse
model_size
candidate_format
```

Avoid using `module_name` as a raw categorical feature in the first version, because it can memorize specific layers rather than learning transferable structure.

## Evaluation

Minimum useful checks:

1. Leave-one-seed-out within 0.5B.
2. Train on 0.5B perturbation rows, test ranking on the 7B targeted panel.
3. Compare against fixed thresholds.
4. Compare against pure outlier ranking.
5. Report top-k precision policy overlap with validated safe modules.

Useful metrics:

```text
AUROC for safe/unsafe classification
Spearman correlation with abs perturbation delta
precision@k for safe low-risk module selection
recall of high-risk module avoidance
```

## Assignment Optimizer

After predicting risk, choose a policy with a budgeted optimizer:

```text
maximize estimated_memory_savings(policy)
subject to predicted_quality_risk(policy) <= epsilon
```

For the first implementation, use a greedy ranker:

```text
sort candidates by predicted risk ascending
add candidates until risk budget is exhausted
```

Later, replace with knapsack or integer programming if real per-module cost estimates are available.

## Expected Files

```text
experiments/h7-precision-predictor/
  protocol.md
  code/
    build_precision_dataset.py
    train_precision_predictor.py
    select_policy_from_predictions.py
  results/
    precision_dataset.csv
    predictor_metrics.json
    selected_policy.json
  analysis.md
```

## First Command Shape

After implementing the dataset builder, the first run should look like:

```bash
python experiments/h7-precision-predictor/code/build_precision_dataset.py \
  --h6-results experiments/h6-adaptive-precision-assignment/results \
  --output experiments/h7-precision-predictor/results/precision_dataset.csv
```

Then:

```bash
python experiments/h7-precision-predictor/code/train_precision_predictor.py \
  --input experiments/h7-precision-predictor/results/precision_dataset.csv \
  --target abs_perturbation_delta \
  --eval-mode leave_one_seed_out \
  --output experiments/h7-precision-predictor/results/predictor_metrics.json
```

## Success Criterion

The predictor is worth adding to the paper if it beats fixed thresholds and pure outlier ranking on at least one meaningful held-out setting, especially 0.5B-to-7B transfer.

If it does not beat pure outlier ranking, that is still useful: the paper should state that activation outlier ranking is sufficient for the current evidence, and learned prediction remains future work.
