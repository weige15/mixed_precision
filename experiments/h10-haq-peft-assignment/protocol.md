# H10 Protocol: HAQ-Style Backend-Aware Precision Assignment for PEFT

## Research Question

Can the HAQ principle be adapted to PEFT by estimating Transformer module precision risk from cheap probes, then solving a backend-aware precision assignment problem over actions the PEFT stack can actually implement?

## Motivation

HAQ showed that mixed-precision assignment should be hardware-aware rather than based on uniform bitwidths or proxy compute metrics. PEFT changes the search space. The decisions are no longer generic CNN layer bitwidths; they are module/action/backend choices such as QLoRA/NF4, bf16 projection rescue, bitsandbytes 8-bit, adapter dtype, or optimizer-state precision.

The project already has the two ingredients needed for a first PEFT adaptation:

- H7: cheap calibration and perturbation probes that estimate module precision risk.
- H8: a hardware-backed selective-rescue path that starts from QLoRA/NF4 and reloads selected projection modules in bf16.

H10 connects them with an explicit action table and assignment solver.

## Locked Hypothesis

A grouped precision-assignment solver using H7/H8 risk estimates and measured backend costs will select a QLoRA/NF4 plus bf16 projection-rescue policy that preserves most of QLoRA's memory savings while improving quality versus blanket QLoRA, under the same 1% eval-loss gate used by H6-H8.

## Confirmatory Prediction

On Llama-3.1-8B LoRA with the lab RTX 3090 setting already used by H8:

- selected policy final eval loss remains within 1% of matched bf16,
- selected policy improves final eval loss versus blanket QLoRA on each matched seed or on the seed-aggregated mean,
- selected policy retains at least 20% peak-memory saving versus matched bf16,
- selected policy has no additional loss spikes or NaN/Inf events versus blanket QLoRA,
- solver output is frozen before any new validation run.

If H10 only reconstructs the already observed H8 policy from existing data, classify it as a method-assembly result rather than a new empirical result.

## Decision Sites

Use grouped decisions first, not per-module free search:

| Group | Baseline action | Candidate action |
|---|---|---|
| all projection modules | QLoRA/NF4 | keep QLoRA/NF4 |
| high-risk rescued projections | QLoRA/NF4 | reload selected projection weights in bf16 before LoRA wrapping |
| norm/logit paths | existing backend default | no-op unless feasibility says otherwise |
| adapters | bf16/default PEFT dtype | optional future action |
| optimizer states | default optimizer precision | optional future action |

The first solver must choose exactly one action per group and reject infeasible backend rows.

## Input Table Schema

The action table is one row per `(model, group, action, backend, hardware)` candidate:

```text
model_name
group_name
module_names
candidate_action
backend
hardware_label
backend_feasible
predicted_quality_risk
predicted_instability_risk
quality_recovery_vs_lowbit
memory_delta_gib_vs_bf16
throughput_delta_pct_vs_bf16
source_artifact
notes
```

For cost minimization, lower `memory_delta_gib_vs_bf16` is better. For rescue from low-bit, higher `quality_recovery_vs_lowbit` is better.

## Solver Objective

Primary H10 objective:

```text
minimize      predicted_quality_risk - alpha * quality_recovery_vs_lowbit
subject to    backend_feasible = true
              predicted_quality_risk <= 0.01
              predicted_instability_risk <= tau
              memory_delta_gib_vs_bf16 <= -memory_saving_target
              exactly one action selected per group
```

Default constraints:

```text
epsilon = 0.01
tau = 0
memory_saving_target = 20% of bf16 peak memory, represented as a negative GiB delta
```

The first implementation can use exhaustive grouped enumeration because the candidate table is small. Greedy or integer programming can be added later if the action space grows.

## Metrics

Primary:

- final validation/eval loss delta versus matched bf16,
- final validation/eval loss improvement versus blanket low-bit baseline,
- peak CUDA memory delta versus matched bf16,
- backend feasibility.

Secondary:

- train tokens/sec delta,
- loss-spike count,
- NaN/Inf count,
- max gradient norm,
- selected-policy trace.

## Artifacts

Expected files:

```text
experiments/h10-haq-peft-assignment/results/action_table.example.csv
experiments/h10-haq-peft-assignment/results/action_table.csv
experiments/h10-haq-peft-assignment/results/selected_policy.json
experiments/h10-haq-peft-assignment/results/solver_trace.json
```

The generated action table should be preferred over the example table once
`code/build_action_table.py` can reconstruct it from matched H8 artifacts. The
example table remains as a smoke-test fixture for the solver.

## Interpretation Rules

- If the policy is not backend-feasible, H10 is not supported even if its predicted quality is good.
- If the policy preserves quality but does not retain memory saving, it is a sensitivity result, not a HAQ-style hardware-aware result.
- If the policy retains memory but fails the 1% quality gate, it is a resource-only low-bit baseline, not a successful precision assignment.
- If the solver reconstructs H8 from existing data, present the result as evidence that the H8 selective-rescue result can be expressed as a general assignment problem.
