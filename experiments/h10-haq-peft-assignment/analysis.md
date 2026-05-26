# H10 Analysis

## 2026-05-26 Generated H8 Action Table

H10 now has the first generated HAQ-for-PEFT action table. The generator:

- reads `experiments/h8-hardware-aware-precision-search/results/llama_h8_metrics_summary.json`,
- filters to matched `rtx3090-lab`, 500-step, 100-eval-batch comparisons,
- aggregates the three matched seeds for `qlora_nf4` and `h8_rescue`,
- emits grouped action rows for projection storage and norm/logit defaults.

Generated rows:

| Group | Action | Predicted quality risk | Memory delta vs bf16 | Throughput delta vs bf16 |
|---|---|---:|---:|---:|
| projection_storage | blanket_qlora_nf4 | 0.00798336 | -5.391328 GiB | -19.698277% |
| projection_storage | qlora_nf4_bf16_projection_rescue | 0.00681742 | -5.105074 GiB | -19.174164% |
| norm_logits | backend_default | 0.00000000 | 0.000000 GiB | 0.000000% |

Under the default solver constraints:

```text
epsilon = 0.01
tau = 0
max_memory_delta_gib = -4.0
alpha = 1.0
```

the solver selects:

- `norm_logits/backend_default`
- `projection_storage/qlora_nf4_bf16_projection_rescue`

Interpretation: the H8 result is now represented as a generated backend-aware
precision-assignment table rather than a hand-written example. This still
reconstructs existing evidence rather than adding a new training result, but it
is meaningful progress toward the HAQ adaptation because the table/solver
interface is now driven by project artifacts.

## 2026-05-26 Selector-Aware Planning Table

Added `code/build_selector_action_table.py` to connect the generated H8 cost
table with the H10 precision-risk selector branch. This is explicitly a
planning bridge, not a new empirical result.

The script uses:

- H8 matched RTX 3090 500-step QLoRA versus rescue measurements as the backend
  cost/recovery model.
- `rescue_selector_evaluation_llama31_8b.json` as candidate top-k rescue sets.
- selector `unsafe_recall_at_k` to scale the measured H8 top-4 quality recovery.

With the target perturbation upper-bound selector included, the solver selects:

- `norm_logits/backend_default`
- `projection_storage/qlora_nf4_bf16_rescue_oracle_perturbation_upper_bound`

This policy rescues:

- `layers.31.mlp.up_proj`
- `layers.31.mlp.gate_proj`
- `layers.2.mlp.down_proj`
- `layers.30.mlp.gate_proj`

Its predicted quality risk is `0.00681742`, matching the aggregate H8
selective-rescue risk, with predicted memory delta `-5.105074 GiB` versus bf16.

Without the perturbation upper-bound row, all non-oracle selectors currently
recover only one of three unsafe Llama projection candidates at `k=4`. The
solver therefore chooses the first tied cheap selector,
`activation_outlier_rescue`, with predicted quality risk `0.00759471`. This is
still inside the 1% quality-risk gate and retains the memory-saving constraint,
but it has much weaker expected quality recovery than the perturbation-selected
policy.

Interpretation:

- The HAQ-for-PEFT table can now represent different rescue selector policies,
  not only a single aggregate projection-storage action.
- The current evidence argues against presenting pure cross-model prediction as
  the main H10 method.
- The defensible method is target calibration plus targeted perturbation labels,
  followed by backend-aware rescue assignment.
