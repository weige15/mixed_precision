# H10 Analysis

## 2026-05-27 Scope Correction

This branch is now archived as PEFT-side exploratory evidence. It should not be
treated as the main H10 direction.

The project plan's "post-training quantization" target means PTQ for an already
trained model before inference deployment, and original HAQ is also an
inference-oriented hardware-aware bitwidth search. Therefore the active H10
direction moves to:

```text
experiments/h10-inference-ptq-assignment/
```

The selector work below remains useful as evidence that calibration and
perturbation probes can rank precision-sensitive modules, but final H10 claims
should be made on inference workloads and backend-real PTQ policies.

## 2026-05-26 Bootstrap

This was the original PEFT-side bootstrap before the 2026-05-27 scope
correction. It should now be read as historical context, not as the active H10
plan. The research story at that time was:

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

At that time, the intended framing was HAQ-inspired PEFT precision assignment,
not "HAQ for Transformers." After the 2026-05-27 scope correction, this framing
is archived because the active H10 direction follows the original HAQ inference
setting more closely.

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

This sharpened the older PEFT-side direction. A learned cross-model predictor is not yet a
strong enough main claim. The safer method claim is **short target calibration
plus targeted perturbation labels for conservative rescue selection**, with
learned prediction treated as an auxiliary baseline until more labeled models
exist. In other words, H10 should emphasize measured pre-training precision
checks over black-box learned transfer if this PEFT branch is revisited.
