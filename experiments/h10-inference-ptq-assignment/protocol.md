# H10 Protocol: HAQ-Style Mixed-Precision PTQ for LLM Inference

## Research Question

Can the HAQ principle be adapted to LLM inference by using short post-training
calibration and perturbation probes to choose backend-feasible mixed-precision
policies that improve the latency-memory-quality trade-off versus global
bf16/fp16 or uniform quantization baselines?

## Motivation

The project plan frames the mixed-precision branch around post-training
quantization, weight quantization, activation-aware normalization, and precision
search for different layers. In that context, "post-training" means PTQ for an
already trained model before inference deployment. It should not be conflated
with PEFT fine-tuning or post-hoc analysis of LoRA training runs.

Original HAQ is also inference-oriented: it searches hardware-aware bitwidth
assignments for a trained model. Therefore H10 should align with inference-side
mixed-precision PTQ. The PEFT/QLoRA selective-rescue artifacts from H8 and the
earlier H10 branch remain useful evidence about calibration signals, but they
are no longer the main H10 target.

## Relationship to H7-H9

- H7 provides cheap sensitivity signals: calibration outliers, fake-quantization
  error, and one-module perturbation deltas.
- H8 provides a PEFT-side backend-real selective-rescue result, now treated as a
  side branch rather than the main HAQ-aligned contribution.
- H9 provides inference-side backend measurements through vLLM: model dtype,
  quantization backend, KV-cache dtype, prefill latency, decode throughput,
  memory, and quality gates.
- H10 should join H7-style sensitivity estimation with H9-style backend cost
  measurement in an inference PTQ assignment table and solver.

## Locked Hypothesis

A backend-aware mixed-precision PTQ solver can select an implementable LLM
inference policy that passes a 1% quality-degradation gate while improving at
least one deployment metric, such as peak memory, prefill latency, decode
throughput, or long-context KV-cache cost, relative to matched bf16/fp16
serving baselines.

## Policy Space

Each policy is a concrete inference deployment configuration, not a LoRA
training recipe:

```text
policy = {
  model_name,
  weight_precision_or_backend,
  activation_quantization_policy,
  kv_cache_dtype,
  attention_backend_or_runtime_mode,
  layer_or_group_precision_overrides,
  hardware_label
}
```

Initial backend-real candidates should reuse H9 infrastructure:

- bf16 default serving,
- fp16 default serving,
- FP8 KV-cache variants when supported by vLLM and the target GPU,
- bitsandbytes weight-only quantization when vLLM can instantiate it,
- torchao/AWQ/GPTQ/Marlin candidates only when compatible artifacts and local
  backend support are available,
- layer/group precision overrides only if the serving backend can express them
  without custom unsupported hooks.

## Input Table Schema

The action table is one row per feasible or attempted inference action:

```text
model_name
group_name
candidate_action
backend
hardware_label
backend_feasible
calibration_signal
perturbation_risk
predicted_quality_risk
memory_delta_gib_vs_bf16
prefill_latency_delta_pct_vs_bf16
decode_tokens_per_sec_delta_pct_vs_bf16
kv_cache_memory_delta_gib_vs_bf16
source_artifact
failure_reason
notes
```

Unsupported rows should stay in the table with `backend_feasible = false`; they
are systems evidence and prevent overclaiming.

## Solver Objective

The first solver can use grouped enumeration because the candidate space is
small:

```text
minimize      predicted_quality_risk
secondary     deployment_cost(policy)
subject to    backend_feasible = true
              predicted_quality_risk <= 0.01
              at least one deployment metric improves versus matched bf16/fp16
              policy is frozen before final quality validation
```

`deployment_cost(policy)` should be reported as a vector, not collapsed into a
single number by default:

```text
memory, prefill latency, decode throughput, KV-cache memory, failure risk
```

This keeps the result compatible with a Pareto-frontier interpretation.

## Metrics

Primary:

- held-out prompt NLL, perplexity, or task accuracy versus matched bf16,
- prefill latency for long-prompt/short-generation workloads,
- decode output tokens/sec for short-prompt/long-generation workloads,
- total generated tokens/sec for mixed workloads,
- peak CUDA memory and long-context KV-cache memory,
- backend feasibility and failure reason.

Secondary:

- load-time memory,
- post-run reserved memory,
- quality agreement or logprob agreement diagnostics,
- solver trace and selected-policy rationale.

## Decision Rules

H10 is supported if a backend-real inference policy is non-dominated against
matched bf16/fp16 defaults and passes the quality gate.

H10 is partially supported if the table/solver infrastructure works but the
available backend policies are dominated or only improve one metric at an
unacceptable cost.

H10 is not supported if the selected policy depends on fake quantization hooks,
unsupported backend flags, or PEFT training behavior rather than inference
deployment evidence.

## Claim Boundary

This H10 is HAQ-inspired because it adapts HAQ's hardware-aware precision
assignment principle to LLM inference PTQ. It is not a PEFT precision-assignment
claim. PEFT/QLoRA rescue results may be cited as prior sensitivity evidence, but
the main H10 contribution must be evaluated on inference workloads.
