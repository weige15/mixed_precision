# H7 Protocol: Precision Assignment Predictor

## Question

Can the H6 calibration artifacts be turned into an explicit predictor that ranks module-format precision risk better than fixed thresholds?

## Motivation

H6 showed that activation outlier signals and perturbation deltas can identify low-risk fake-int8 modules before LoRA training. H7 tests the next abstraction: represent each candidate module as a feature row, train a simple predictor for perturbation sensitivity, and use the predictor to select a frozen policy.

## Confirmatory Test

This first H7 implementation is a post-hoc predictor over existing H6 artifacts. It does not run new model training.

The locked evaluation checks are:

1. Leave-one-seed-out prediction over rows with perturbation labels.
2. 0.5B-to-7B transfer: train on Qwen2.5-0.5B perturbation rows and test on the targeted Qwen2.5-7B panel.
3. Compare the learned predictor against fixed thresholds and pure outlier ranking.

## Dataset

Each row is one `(model, seed, module, candidate_format)` pair. The initial candidate format is fake int8 output quantization because that is what H6 perturbation probes measured.

Core columns:

```text
model_name
model_size_b
seed
module_name
layer_idx
normalized_depth
module_role
module_leaf
candidate_format
activation_outlier_score
input_outlier_score
output_outlier_score
int8_rel_mse
output_int4_rel_mse
perturbation_delta
abs_perturbation_delta
safe_label
label_source
```

The initial safe label is:

```text
safe_label = abs_perturbation_delta <= 0.005
```

Rows without perturbation labels are kept for future policy selection but excluded from predictor training.

## Predictor

The first predictor is intentionally simple:

- Ridge regression for absolute perturbation delta.
- Logistic regression for the safe/unsafe label when both classes are present.

Features:

```text
model_size_b
layer_idx
normalized_depth
log1p activation_outlier_score
log1p int8_rel_mse
log1p output_int4_rel_mse
module_role
module_leaf
candidate_format
```

The predictor must not use raw `module_name` in the first version, because that would encourage memorizing specific layers.

## Success Criterion

H7 is useful enough to report if it beats either fixed thresholds or pure outlier ranking on a held-out setting. If it does not, the result is still useful: the paper should keep the simpler claim that activation-outlier ranking is currently sufficient.

