# H10 Analysis

## 2026-05-26 Bootstrap

H10 becomes the main post-training branch after H8 and H9. The research story is
now:

- H6: short calibration and perturbation probes expose module precision
  sensitivity.
- H7: simple predictors can be trained from those module rows, but outlier-only
  ranking remains a strong baseline.
- H8: selective rescue from QLoRA/NF4 is backend-real and preserves most memory
  savings while modestly improving quality versus blanket QLoRA.
- H9: inference-side global vLLM knobs are too coarse to be the main HAQ-style
  contribution.

The immediate H10 task is to turn the existing H6/H7/H8 artifacts into a
decision-ready module-rescue evaluation: aggregate labels across seeds, compare
rescue selectors at equal budget, then run only the best non-oracle policies on
GPU.

H10 should be reported as HAQ-inspired PEFT precision assignment, not "HAQ for
Transformers." The transferable principle is constrained hardware-aware
precision assignment; the action space, labels, and objective are different from
CNN inference quantization.

## 2026-05-26 Seed-Aggregated Selector Screen

Implemented the first CPU-only H10 artifact layer:

- `code/build_seed_aggregated_dataset.py`
- `code/evaluate_rescue_selectors.py`

The aggregated dataset collapses 114 seed-level perturbation rows from H7 into
38 module-level rows:

| Model | Module rows | Projection rows | Unsafe-any-seed rows |
|---|---:|---:|---:|
| Qwen/Qwen2.5-0.5B | 10 | 7 | 5 |
| Qwen/Qwen2.5-7B | 14 | 10 | 10 |
| meta-llama/Llama-3.1-8B | 14 | 10 | 6 |

For rescue selection, the projection-only candidate sets are smaller:

| Model | Projection candidates | Unsafe-any-seed projections |
|---|---:|---:|
| Qwen/Qwen2.5-7B | 10 | 6 |
| meta-llama/Llama-3.1-8B | 10 | 3 |

The first selector screen is mixed and useful. On Qwen2.5-7B, activation
outlier ranking and the role-prior selector match the perturbation upper bound
at `k=4`, capturing four unsafe projection modules. On Llama-3.1-8B, the same
cheap selectors capture only one of the three unsafe projection modules; the
cross-model learned predictors also capture only one. The only selector that
finds all three unsafe Llama projections is the target perturbation upper bound,
which selects:

- `layers.31.mlp.up_proj`
- `layers.31.mlp.gate_proj`
- `layers.2.mlp.down_proj`
- `layers.30.mlp.gate_proj`

This sharpens the H10 direction. A learned cross-model predictor is not yet a
strong enough main claim. The safer method claim is **short target calibration
plus targeted perturbation labels for conservative rescue selection**, with
learned prediction treated as an auxiliary baseline until more labeled models
exist. In other words, H10 should emphasize measured pre-training precision
checks over black-box learned transfer.
