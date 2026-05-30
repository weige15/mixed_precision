# Detailed Design

## Purpose

This document turns `qaq/doc/proposal.md` into an implementation-oriented design for a project-local QAQ prototype. The prototype explores query-adaptive mixed-precision quantization for LLM inference: one stored model representation should support different reconstructed bit widths per query, then report quality, memory, latency, and routing behavior against fixed-precision baselines.

The design keeps the proposal's first-deliverable constraint: this is a readable PyTorch/Transformers research prototype under `qaq/`, not a custom CUDA, Triton, vLLM, or production kernel project.

## Source Proposal Summary

The proposal defines QAQ as a research prototype with five major implementation sections:

1. Bit-plane weight representation.
2. Static mixed-precision baseline.
3. Query-conditioned router.
4. Dynamic loader.
5. Evaluation harness.

The proposal assumes root `requirements.txt` remains the dependency source, selected weights are quantized to a maximum of 8 bits, reconstruction supports candidate widths such as 2, 4, 6, and 8 bits, routing starts at block or module-group granularity, and results are written as JSONL plus summary JSON in the style of the existing experiment directories.

The existing high-level design records `Llama-3.1-8B-Instruct` as the target model, defaulting to `meta-llama/Llama-3.1-8B-Instruct` when a Hugging Face identifier is needed. The detailed design treats that as the target for final evidence, while keeping tiny-model or toy-tensor modes for independent correctness tests.

## Design Goals

- Provide deterministic bit-plane quantization and reconstruction utilities for selected LLM weights.
- Compare query-adaptive routing against fixed 8-bit, fixed 4-bit, hand-authored mixed, no-router, random-router, and oracle-label baselines.
- Keep policy decisions keyed by stable group ids rather than raw tensor traversal order.
- Make router behavior inspectable through saved features, labels, predictions, selected bit widths, and distribution summaries.
- Separate algorithmic QAQ results from optional CPU-to-GPU loading results so synchronous transfer overhead is not hidden.
- Reuse repository conventions: simple `argparse` entrypoints, root `requirements.txt`, explicit seeds, explicit `hardware_label`, `metrics.jsonl`, and `summary.json`.

## Non-Goals

- No custom CUDA, Triton, vLLM backend, or production serving scheduler in the first version.
- No per-weight or per-channel query routing in the first version.
- No latency-improvement claim for dynamic loading before measurement.
- No replacement of existing H9/H10 artifact-backed PTQ experiments; QAQ is a separate bit-plane reconstruction prototype.
- No silent choice of final hardware, dataset, or quality tolerance while those remain open.

## Architecture Overview

The implementation should live under `qaq/` and follow the repository's existing experiment style:

```text
qaq/
  code/
    bitplanes.py
    policies.py
    model_adapter.py
    router.py
    oracle_labels.py
    evaluate.py
    dynamic_loader.py
  tests/
    test_bitplanes.py
    test_policies.py
    test_router.py
  doc/
    proposal.md
    high-level-design.md
    detailed-design.md
  results/
```

This file layout is a design target, not a statement that the files already exist.

Runtime flow:

1. Load configuration from CLI arguments and serialize it into the output directory.
2. Load tokenizer/model and build a selected-module inventory.
3. Quantize selected weight tensors to a maximum bit width and split integer values into ordered bit planes.
4. Store bit-plane tensors and metadata in memory, and optionally persist them for reuse.
5. Build static policy mappings or query-conditioned policy mappings.
6. Reconstruct selected weights from top-k most significant bit planes.
7. Run model inference with reconstructed weights.
8. Record per-sample quality, latency, memory, and selected-bit metadata.
9. Aggregate `summary.json` and optional artifacts such as `policy_trace.json`, router labels, and router predictions.

## Module Designs

### Bit-plane weight representation

#### Responsibility

This module owns quantizing selected floating-point weight tensors to a configured maximum bit width and splitting the resulting integer representation into ordered bit planes. It also owns metadata required for deterministic reconstruction.

It does not choose a query policy, train a router, run model evaluation, or claim hardware speedups.

#### Inputs and Outputs

Inputs:

- Floating-point weight tensors from selected linear modules.
- Tensor names, module names, layer indices, and group ids from the model adapter.
- Quantization configuration: `max_bits`, allowed bit widths, quantization scheme, target storage dtype, and device placement.

Outputs:

- `BitPlaneTensor` records keyed by tensor name.
- Quantization metadata including shape, scale, zero point or symmetric scale, max bit width, plane order, source dtype, storage dtype, device placement, and group id.
- Optional serialized bit-plane artifact for reuse by evaluation runs.

Suggested data contracts:

```text
tensor_name
module_name
shape
source_dtype
quantization_scheme
scale
zero_point
max_bits
plane_order
storage_dtype
device_placement
group_id
planes
```

#### Internal Design

The first implementation should use symmetric per-tensor quantization unless the implementation owner confirms an asymmetric scheme. Symmetric quantization keeps toy-tensor tests and metadata easier to audit:

1. Convert source tensor to `float32` for quantization math.
2. Compute `scale = max(abs(weight)) / (2 ** (max_bits - 1) - 1)` with a zero-tensor guard.
3. Quantize to a signed integer range.
4. Convert the signed values into a representation that can be split into ordered bit planes.
5. Store planes from most significant to least significant so "top-k" reconstruction is unambiguous.
6. Preserve enough signedness metadata to reconstruct negative values correctly.

The module should expose pure tensor-level functions before model-level wrappers:

```text
quantize_tensor_to_bitplanes(weight, config) -> BitPlaneTensor
reconstruct_tensor(record, bit_width, dtype, device) -> torch.Tensor
estimate_storage_bytes(record, bit_width=None) -> int
```

Exact reconstruction at `max_bits` means exact reconstruction of the quantized/dequantized value, not exact recovery of the original float tensor.

#### Dependencies

- PyTorch tensor operations.
- Model inventory records from the model adapter.
- Root `requirements.txt`; no new package is required for the first version.

#### Failure Handling

- Reject unsupported `bit_width` values that are not in `allowed_bit_widths` or exceed `max_bits`.
- Reject unsupported quantization schemes with a clear `ValueError`.
- Preserve deterministic behavior for zero tensors by using scale `1.0` and all-zero integer payloads.
- Validate that reconstructed tensor shape matches metadata before returning it.
- If a tensor contains NaN or Inf, fail during quantization and report the tensor name.

#### Independent Test Plan

- Test signed and unsigned toy tensors at 2, 4, 6, and 8 bits.
- Test exact reconstruction of quantized values at `max_bits`.
- Test zero tensors, single-value tensors, non-contiguous tensors, and shape preservation.
- Test that lower-bit reconstruction uses only the top-k most significant planes.
- Test storage-byte accounting without loading a model.

No repo-level test command exists yet. Once tests are added, this module should be runnable independently with either a direct script-level test command or the agreed project test runner.

#### Open Questions

- Should the first implementation be symmetric per-tensor quantization, asymmetric affine quantization, or both?
- Should bit planes be persisted as unpacked boolean or integer tensors first, or should packed storage be required in the first deliverable?

### Static mixed-precision baseline

#### Responsibility

This module represents fixed precision policies and hand-authored mixed policies so that query-adaptive QAQ has stable baselines. It validates policy coverage over group ids and provides group-to-bit-width mappings for reconstruction.

It does not train or infer query-conditioned decisions.

#### Inputs and Outputs

Inputs:

- Module inventory with group ids, module roles, layer indices, and tensor names.
- Policy definitions from built-in names or JSON files.
- Allowed bit widths from quantization configuration.

Outputs:

- Validated `PrecisionPolicy` records.
- A complete `group_id -> bit_width` mapping.
- Policy metadata for result files.

Suggested policy contract:

```text
policy_name
model_name
group_granularity
allowed_bit_widths
default_bit_width
group_bit_widths
source
description
```

#### Internal Design

Provide built-in policies first:

- `static_8bit`: every selected group uses 8-bit reconstruction.
- `static_4bit`: every selected group uses 4-bit reconstruction.
- `mixed_attention_high`: attention groups use a higher bit width than MLP groups when those roles are available.
- `random_router_baseline`: seeded random group bit widths for ablation, implemented through the same policy interface but clearly marked as random.
- `oracle_policy`: loaded from oracle-label artifacts for a single sample or aggregate evaluation path.

The policy module should never inspect raw weights. It only consumes inventory and produces mappings.

#### Dependencies

- Module inventory records from the model adapter.
- Bit-width constraints from bit-plane configuration.
- JSON read/write from the standard library.

#### Failure Handling

- Reject policies that reference unknown group ids.
- Reject missing group assignments unless `default_bit_width` is set.
- Reject bit widths not supported by the bit-plane store.
- Save the fully expanded policy mapping in result artifacts so later analysis does not depend on default expansion logic.

#### Independent Test Plan

- Test built-in fixed policies on a synthetic inventory.
- Test hand-authored mixed policies with attention and MLP roles.
- Test rejection of unknown groups and unsupported bit widths.
- Test deterministic random baseline with a fixed seed.
- Test JSON round-trip of policy definitions without model loading.

#### Open Questions

- What should the first hand-authored mixed policy be if group granularity is transformer-layer only and attention/MLP roles are not separate?
- Should static mixed policies reuse H10 late-MLP patterns for Llama-3.1-8B-Instruct, or remain architecture-neutral until the first inventory is generated?

### Query-conditioned router

#### Responsibility

This module computes query features, trains or loads a lightweight router, and predicts a bit-width policy per query. It also supports oracle-label generation as the supervised bootstrap path described by the proposal.

It does not own bit-plane storage or low-level reconstruction.

#### Inputs and Outputs

Inputs:

- Tokenized prompt and attention mask.
- Optional prompt metadata supplied by the evaluation harness.
- Optional early hidden states or activation summaries when enabled.
- Module inventory and allowed group ids.
- Oracle labels or candidate-policy score tables.

Outputs:

- Feature records.
- Oracle label records.
- Router model artifact and router config.
- Per-query predicted policy mapping.
- Router trace entries for evaluation outputs.

Suggested router record:

```text
sample_id
prompt_metadata
feature_schema_version
features
candidate_scores
oracle_label_or_predicted_policy
selected_group_bit_widths
expected_cost
quality_metric
router_confidence
```

#### Internal Design

The router path should be staged:

1. Token-only features: input length, attention-mask length, and optional prompt/task metadata.
2. Model-derived features: pooled embedding or first-token hidden state from an early pass when enabled.
3. Activation summaries: outlier or saturation-style summaries from selected blocks only after the basic path works.
4. Oracle labels: evaluate candidate policies per calibration sample and choose the lowest-cost policy that stays within the configured quality tolerance.
5. Supervised router: train a lightweight classifier or multilabel predictor from features to policy ids or group bit-width choices.
6. Distillation path: add KL divergence or next-token cross entropy plus cost penalty only after supervised labels are validated.

For the first implementation, policy-id classification is simpler and more auditable than predicting every group independently. Group-wise prediction can be added after the candidate policy grid and label quality are understood.

#### Dependencies

- PyTorch for lightweight router models.
- Transformers tokenizer/model outputs when model-derived features are enabled.
- Evaluation harness for oracle scoring.
- Precision policy module for candidate policy definitions.

#### Failure Handling

- Save feature schema versions and reject incompatible router artifacts.
- Validate that router predictions expand to legal bit widths for all required groups.
- If model-derived feature extraction fails because the target model is unavailable, allow token-only smoke tests but mark them as non-target.
- If no candidate policy satisfies the quality tolerance during oracle labeling, emit the highest-precision candidate with a `tolerance_satisfied=false` flag rather than hiding the failure.

#### Independent Test Plan

- Test token-only feature extraction from synthetic tokenized batches.
- Test oracle label selection from synthetic candidate scores, including no-candidate-satisfies cases.
- Test router training and prediction on a tiny synthetic dataset.
- Test policy expansion and legality checks from predicted labels.
- Test router artifact save/load with feature schema validation.

#### Open Questions

- Should the first router target policy-id classification or direct group-wise bit-width prediction?
- What group granularity should the first router use: transformer layer, attention vs MLP group, or individual linear module?
- What quality tolerance defines "minimal sufficient precision" for oracle labels?
- Which calibration dataset should be the default for the target model?

### Dynamic loader

#### Responsibility

This optional module keeps bit planes in CPU memory and materializes only selected planes on GPU for each query. It measures the memory-latency trade-off separately from the core bit-plane and router algorithm.

It does not change router decisions or policy semantics.

#### Inputs and Outputs

Inputs:

- Bit-plane store records.
- Per-query or static precision policy mapping.
- Target device and dtype.
- Loader configuration controlling synchronous transfer and measurement.

Outputs:

- Reconstructed tensors on the target device.
- Transfer timing records.
- CPU resident storage and GPU materialization metrics.
- Loader-specific summary fields.

Suggested loader metrics:

```text
sample_id
policy_name
cpu_plane_bytes
gpu_materialized_bytes
transfer_ms
reconstruction_ms
total_loader_ms
device
```

#### Internal Design

The first dynamic loader should be synchronous and explicit:

1. Keep canonical bit-plane records on CPU.
2. For each selected tensor, transfer only planes needed by the selected bit width.
3. Reconstruct on the target device.
4. Return reconstructed tensors through the same reconstruction interface used by non-loader evaluation.
5. Record transfer and reconstruction timings separately.

This module should be behind a configuration flag such as `--dynamic-loader`. Evaluation summaries must keep loader-enabled metrics separate from non-loader metrics.

#### Dependencies

- Bit-plane representation and reconstruction functions.
- PyTorch CUDA APIs when CUDA is available.
- Evaluation harness measurement utilities.

#### Failure Handling

- If CUDA is unavailable, reject GPU dynamic loading and allow CPU-only dry-run loader tests.
- If a transfer fails due to memory pressure, record the failed sample and stop the run with a clear error.
- Always report latency overhead fields when dynamic loading is enabled.
- Do not merge dynamic-loader summaries into core QAQ summaries without a loader mode label.

#### Independent Test Plan

- Test CPU-only materialization on toy bit-plane records.
- Test device placement validation without requiring the target model.
- Test transfer metric fields with mocked or CPU fallback timers.
- Test that dynamic loading returns the same reconstructed tensor values as the non-loader path for the same bit width.

#### Open Questions

- Should persisted CPU storage be implemented in the first version, or is in-memory CPU storage enough for the first dynamic-loader measurement?
- Which hardware label is the first target for meaningful dynamic-loader evidence?

### Evaluation harness

#### Responsibility

This module coordinates model loading, dataset loading, bit-plane preparation, policy selection, inference, metrics, and result writing. It is the only module that should know about full experiment runs.

It does not own quantization math, router training internals, or policy validation beyond calling the owning modules.

#### Inputs and Outputs

Inputs:

- CLI configuration: model name or local path, dataset name, split, sample limits, sequence length, seed, dtype, group granularity, policy names, router artifact path, hardware label, and output directory.
- Built-in or JSON policy definitions.
- Optional router labels or router artifact.

Outputs:

- `metrics.jsonl`.
- `summary.json`.
- Optional `policy_trace.json`.
- Optional `features.jsonl`, `oracle_labels.jsonl`, `candidate_scores.jsonl`, and router artifact files.
- Copied or serialized run configuration.

Minimum summary fields:

```text
model_name
model_path_or_id
dataset_name
dataset_split
seed
hardware_label
group_granularity
allowed_bit_widths
policies
quality_metrics
latency_metrics
memory_metrics
selected_bit_distribution
dynamic_loader_enabled
errors
```

#### Internal Design

The harness should expose separate entrypoint modes rather than one opaque script:

- `prepare-bitplanes`: load model, build inventory, and create bit-plane records.
- `run-static`: evaluate static policies.
- `build-oracle-labels`: score candidate policies per sample.
- `train-router`: train router from features and labels.
- `run-router`: evaluate query-conditioned routing.
- `run-dynamic-loader`: rerun selected policy paths with loader metrics.

For early implementation, these modes can live in one `qaq/code/evaluate.py` script with subcommands. If the script becomes large, split subcommands into separate files while preserving output contracts.

Quality metrics should start with next-token loss or perplexity-style prompt loss because it is repeatable and does not require task-specific answer normalization. Small instruction exact-match probes can be added as secondary evidence, following the H10 task-quality pattern.

Latency metrics should distinguish prefill and decode where feasible. If the prototype only supports end-to-end generation timing at first, the summary must label it as end-to-end timing rather than prefill/decode timing.

#### Dependencies

- PyTorch, Transformers, Datasets, NumPy, Pandas, and tqdm from root `requirements.txt`.
- Bit-plane, policy, router, oracle, model adapter, and optional loader modules.
- Existing repository output conventions from H6/H7/H10.

#### Failure Handling

- Fail fast when the target model cannot be loaded, unless `--tiny-smoke-model` or equivalent smoke configuration is explicitly selected.
- Record model access mode: Hugging Face id, local path, or offline cache.
- Stop if no selected modules are found for quantization.
- Serialize partial metrics only when they are internally valid and mark incomplete runs in `summary.json`.
- Keep hardware-specific results separated by `hardware_label` and output directory.

#### Independent Test Plan

- Test CLI parsing and config serialization without model loading.
- Test dataset formatting on synthetic examples.
- Test metrics aggregation from synthetic `metrics.jsonl`.
- Test summary generation with and without dynamic-loader fields.
- Test an end-to-end toy run with a tiny local model or a mocked model adapter after the core modules exist.

#### Open Questions

- Which exact dataset and split should be the default calibration/evaluation set?
- Should final assignment evidence prioritize prompt NLL/perplexity, instruction exact-match tasks, or another metric?
- Should the first runnable mode support a tiny public model in addition to `Llama-3.1-8B-Instruct` for smoke testing?

## Cross-Module Contracts

### Module Inventory

The model adapter should provide the shared inventory consumed by bit-plane representation, static policies, router grouping, and evaluation:

```text
tensor_name
module_name
module_role
layer_idx
group_id
group_granularity
shape
source_dtype
target_quantized
```

Group ids must be stable across runs for the same model, module filter, and group granularity.

### Bit-Plane Store

The bit-plane store is keyed by tensor name and includes a reverse lookup from group id to tensor names. Router and policy modules must not mutate the store.

### Precision Policy

All static, random, oracle, and router-selected decisions must expand to:

```text
sample_id_or_global
policy_name
group_bit_widths
default_bit_width
allowed_bit_widths
source
```

The reconstruction engine only accepts expanded policies.

### Metrics Records

Every per-sample record written to `metrics.jsonl` should include:

```text
sample_id
policy_name
seed
hardware_label
input_tokens
output_tokens
quality
latency
memory
selected_group_bit_widths
dynamic_loader_enabled
```

Fields that are unavailable in a mode should be present as `null` or omitted consistently and documented in `summary.json`.

## Test Strategy

Testing should progress in layers:

1. Tensor-only tests for bit-plane split, reconstruction, and storage accounting.
2. Synthetic-inventory tests for static policies and group mapping.
3. Synthetic-feature tests for oracle labels and router predictions.
4. Harness aggregation tests using small generated JSONL fixtures.
5. Tiny-model smoke tests for model wrapping and deterministic logits.
6. Target-model experiments for final evidence, separated by hardware label and output directory.

The proposal's validation plan remains the acceptance shape:

- Static 4-bit, static 8-bit, and mixed policies must be compared.
- Router evaluation must use a held-out split after calibration/training.
- Query-adaptive routing must beat at least one static baseline on the chosen assignment metric.
- Dynamic loading, if enabled, must report both memory savings and latency overhead separately.

## Risks and Mitigations

- Python reconstruction may dominate latency. Mitigation: report reconstruction time separately and treat early results as algorithmic evidence.
- Llama-3.1-8B-Instruct access or memory requirements may slow iteration. Mitigation: keep toy-tensor and tiny-model smoke tests clearly labeled, and require final evidence to use the target model or an approved local path.
- Router features from an early model pass may cost more than they save. Mitigation: keep token-only routing as a baseline feature set.
- Oracle label generation may be expensive. Mitigation: start with a small candidate policy grid and calibration subset.
- Ambiguous group granularity could make policy comparisons hard to interpret. Mitigation: serialize inventory, group ids, group granularity, and expanded policies with every run.
- Dynamic loading could reduce memory while increasing latency. Mitigation: keep loader metrics separate and avoid merged claims.

## Open Questions

1. Should final Llama-3.1-8B-Instruct runs use `meta-llama/Llama-3.1-8B-Instruct`, a local checkpoint path, or a pre-approved mirror?
2. Which hardware label is the primary target for final QAQ evidence: CPU-only smoke, RTX 3090, A100, or another machine?
3. What group granularity should be implemented first: transformer layer, attention vs MLP group, or individual linear module?
4. What quality tolerance defines "minimal sufficient precision" during oracle label generation?
5. Which calibration and evaluation dataset should be the default for target-model runs?
6. Should the first router predict a policy id from a candidate grid, or directly predict group-wise bit widths?
7. Should the first quantizer use symmetric per-tensor quantization only, or include asymmetric affine quantization from the beginning?
8. Should persisted bit-plane storage be required in the first deliverable, or can the first version keep bit planes in memory and add persistence later?
