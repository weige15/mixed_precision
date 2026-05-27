# Superseded H10 Progress: HAQ Principle Adapted to PEFT

Date: 2026-05-27

## Scope Status

This memo describes the older PEFT/QLoRA selective-rescue formulation of H10.
It is now superseded as the main H10 direction. The active direction is
HAQ-style mixed-precision post-training quantization for LLM inference:

```text
experiments/h10-inference-ptq-assignment/protocol.md
```

The PEFT results below remain supporting evidence, but H10 should now be framed
around inference deployment, backend-real PTQ actions, and vLLM-style latency,
memory, KV-cache, and quality metrics.

## Bottom Line

The HAQ adaptation is now concrete:

> Instead of RL over CNN layer bitwidths, H10 builds a table of Transformer PEFT
> `(module/group, action, backend, hardware)` candidates, estimates precision
> risk from cheap probes, attaches measured backend cost, and solves a frozen
> constrained assignment problem.

The current best formulation is **target perturbation-guided selective rescue
from QLoRA/NF4**, not pure cross-model prediction.

## What Changed

H8 already showed a backend-real memory-quality trade-off on Llama-3.1-8B:

| Policy | Eval degradation vs bf16 | Peak memory vs bf16 | Throughput vs bf16 |
|---|---:|---:|---:|
| QLoRA/NF4 | +0.798% | -26.70% | -19.70% |
| QLoRA/NF4 + bf16 projection rescue | +0.682% | -25.28% | -19.17% |

H10 now turns that result into an assignment artifact. The generated action
table aggregates matched RTX 3090 500-step runs across seeds 42-44:

| Candidate action | Predicted quality risk | Memory delta vs bf16 | Instability risk |
|---|---:|---:|---:|
| blanket QLoRA/NF4 | 0.00798336 | -5.391 GiB | 0 |
| QLoRA/NF4 + bf16 projection rescue | 0.00681742 | -5.105 GiB | 0 |

Under a 1% quality-risk gate and required memory saving, the solver selects
`qlora_nf4_bf16_projection_rescue`.

## Selector Finding

The selector-aware planning table compares possible top-k rescue selectors.
For Llama-3.1-8B at `k=4`:

| Selector family | Unsafe projection recall | Predicted quality risk |
|---|---:|---:|
| Target perturbation ranking | 3/3 | 0.00681742 |
| Activation outlier | 1/3 | 0.00759471 |
| INT8 MSE | 1/3 | 0.00759471 |
| Role prior | 1/3 | 0.00759471 |
| Cross-model predictors | 1/3 | 0.00759471 |

This is the key interpretive update: cheap signals are useful, but not strong
enough on Llama to replace a short target perturbation check. The defensible
method is:

1. run cheap calibration on the target model,
2. run targeted one-module perturbation probes for plausible rescue candidates,
3. build backend-feasible action rows,
4. solve the constrained assignment,
5. freeze and validate the selected policy.

## Why This Is a PEFT Version of HAQ

HAQ's durable contribution was hardware-aware precision assignment. The PEFT
version changes the objects:

| HAQ Original | H10 PEFT Adaptation |
|---|---|
| CNN layers | Transformer modules/groups |
| Bitwidth actions | PEFT backend actions |
| RL controller | Table solver / constrained assignment |
| Hardware latency feedback | Measured memory, throughput, feasibility |
| Accuracy reward | Predicted/evaluated fine-tuning quality risk |

The RL mechanism is optional. The important idea is the same: precision policy
selection should optimize over actual backend behavior, not global dtype recipes.

## Current Claim Boundary

Supported:

- H10 can express H8 as a backend-aware assignment table.
- Solver output selects selective bf16 rescue from QLoRA/NF4 under the locked
  quality and memory constraints.
- Target perturbation labels are currently the strongest rescue selector.

Not yet supported:

- A fully general optimizer over all Transformer modules and backend formats.
- A pure learned predictor that replaces target perturbation labels.
- A throughput win; the best current result is a memory-quality trade-off.

## Next Best Step

If GPU time is available, validate the perturbation-selected Llama rescue set
from the selector-aware table:

- `layers.31.mlp.up_proj`
- `layers.31.mlp.gate_proj`
- `layers.2.mlp.down_proj`
- `layers.30.mlp.gate_proj`

Otherwise, the next writing step is to turn H10 into the paper's method section:
**calibration and perturbation risk estimation, backend action table, constrained
assignment solver, frozen policy validation**.
