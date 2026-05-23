# H8 Protocol: Hardware-Aware Precision Policy Search

## Question

Can the H6/H7 calibration and perturbation signals be turned into a hardware-aware precision assignment optimizer that selects per-module precision policies with better quality/resource trade-offs than blanket bf16 or blanket QLoRA?

## Motivation

H6/H7 show that short calibration probes can identify modules that tolerate fake low precision while preserving LoRA validation quality. Those results are useful scientifically, but fake-int8 hooks do not create real memory or speed savings. H6.3 shows that blanket QLoRA on Qwen2.5-7B creates a real memory-capacity trade-off on RTX 3090, but it is slower than bf16 and does not use the sensitivity information.

H8 connects these two threads. Instead of demoting a few bf16 modules, H8 starts from a hardware-backed low-bit baseline and selectively rescues high-risk modules or roles to higher precision when predicted quality risk justifies the cost.

## Hypothesis

A hardware-aware policy optimizer can improve the quality side of a low-bit LoRA/QLoRA baseline by rescuing predicted high-risk modules to bf16/fp32, while preserving most of the memory benefit of the low-bit backend.

## Policy Formulation

Each candidate policy chooses a precision/backend state for module groups:

```text
policy[module_or_group] in {low_bit_backend_default, bf16_rescue, fp32_rescue}
```

The first H8 implementation should use coarse groups rather than arbitrary per-module combinations:

- high-risk norm/logit paths
- high-risk early MLP down projections
- selected attention projections
- selected late MLP gate/up projections
- remaining projections

This reduces the search space and avoids pretending that independent single-module perturbations are fully additive.

## Objective

```text
maximize      measured_memory_saving(policy)
secondary     measured_train_tokens_per_sec(policy)
subject to    final_eval_loss_delta <= 1% vs matched bf16
              no added loss spikes or NaN/Inf events
              policy is frozen before training
              all hardware metrics use matched GPU label and microbatching
```

## Predictor Inputs

Reuse and extend H7 features:

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
```

H8 adds backend/cost fields:

```text
backend
format
group_name
predicted_risk
estimated_memory_saving
estimated_throughput_delta
policy_action
```

## First Confirmatory Experiment

Run on one model/hardware pair first:

```text
model: Qwen/Qwen2.5-7B
hardware: lab RTX 3090
training: LoRA/QLoRA, 500 optimizer steps
seeds: start with seed 42, replicate 43/44 only if seed 42 is promising
```

Compare:

1. matched bf16 LoRA baseline
2. blanket QLoRA/NF4 baseline
3. H8 selective rescue policy from QLoRA/NF4

Primary success criterion:

```text
H8 improves eval loss versus blanket QLoRA while retaining most QLoRA memory savings.
```

Do not claim throughput improvement unless measured on matched hardware and microbatching.

## Initial Policy Candidates

Start with policies that are small enough to implement and interpret:

1. `h8_rescue_norm_logits`: low-bit baseline with norm/logit paths kept high precision if backend supports it.
2. `h8_rescue_highrisk_down`: additionally rescue early/high-risk MLP down projections identified by H6/H7 perturbation deltas.
3. `h8_rescue_predictor_topk`: rescue top-k predicted high-risk projection groups under a memory-loss budget.

If backend support cannot express true module rescue without losing QLoRA memory benefits, record that as a systems constraint and keep H8 as a design/prototype branch.

## Decision Rules

H8 is supported if:

- final eval loss is closer to bf16 than blanket QLoRA,
- final eval loss remains inside the 1% gate versus bf16,
- peak memory remains meaningfully below bf16,
- instability events do not increase,
- and all comparisons are matched by hardware label and microbatching.

H8 is not supported if rescue removes most memory savings, introduces large dispatch overhead, or produces no quality improvement over blanket QLoRA.

## Notes

HAQ is relevant as a hardware-aware search template, but its released implementation targets ImageNet CNNs, not Transformers. H8 borrows the constrained search framing, not the code.

