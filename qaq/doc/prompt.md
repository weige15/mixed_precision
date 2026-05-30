# Vibe Coding Prompt: QAQ Query-Adaptive Mixed-Precision Quantization

## Objective

Implement the `qaq/` project-local QAQ prototype end to end. The goal is a readable, testable PyTorch/Transformers research implementation of query-adaptive mixed-precision LLM inference inspired by QAQ: one stored bit-plane model representation should support different reconstructed bit widths per query, then report quality, latency, memory, and routing behavior against fixed-precision baselines.

The implementation must stay a prototype under `qaq/`, not a custom CUDA, Triton, vLLM, or production serving backend. The target model for final evidence is `Llama-3.1-8B-Instruct`, defaulting to `meta-llama/Llama-3.1-8B-Instruct`, while unit tests and smoke paths must run without gated target-model access.

## Inputs

Read these files first:

- `qaq/doc/proposal.md`
- `qaq/doc/high-level-design.md`
- `qaq/doc/detailed-design.md`
- `qaq/doc/tasks/progress.md`
- `qaq/doc/tasks/model-adapter-and-inventory.md`
- `qaq/doc/tasks/bit-plane-weight-representation.md`
- `qaq/doc/tasks/static-mixed-precision-baseline.md`
- `qaq/doc/tasks/query-conditioned-router.md`
- `qaq/doc/tasks/evaluation-harness.md`
- `qaq/doc/tasks/dynamic-loader.md`
- `README.md`
- `requirements.txt`

Use these adjacent implementation references for repository style and output conventions:

- `experiments/h7-precision-predictor/code/inspect_model_modules.py`
- `experiments/h7-precision-predictor/code/train_precision_predictor.py`
- `experiments/h9-transformer-inference-policy-search/code/generate_h9_policies.py`
- `experiments/h10-inference-ptq-assignment/code/build_inference_action_table.py`
- `experiments/h10-inference-ptq-assignment/code/run_h10_task_quality.py`

## Current Implementation

The repository is an existing mixed-precision LLM research workspace using root-level `requirements.txt`; there is no `pyproject.toml`, no repo-level `tests/` directory, and no configured formatter, linter, type checker, or pytest dependency in the discovered environment. Python is available as `python3`; `python`, `pytest`, `ruff`, and `mypy` were not available through the current global interpreter.

The current `qaq/` directory contains only planning docs:

```text
qaq/
  doc/
    proposal.md
    high-level-design.md
    detailed-design.md
    prompt.md
    tasks/
      progress.md
      model-adapter-and-inventory.md
      bit-plane-weight-representation.md
      static-mixed-precision-baseline.md
      query-conditioned-router.md
      evaluation-harness.md
      dynamic-loader.md
```

Create the implementation skeleton described by the design:

```text
qaq/
  code/
    __init__.py
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
    test_model_adapter.py
    test_router.py
    test_dynamic_loader.py
    test_evaluate.py
  results/
```

The root dependency list currently contains `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `numpy`, `pandas`, and `tqdm`. Add only the smallest necessary test dependency if needed, preferably `pytest`, and keep the project requirements-based unless the user later asks for a packaging conversion.

Repository conventions to preserve:

- Use simple `argparse` entrypoints.
- Save run configuration and artifacts explicitly.
- Use `metrics.jsonl` and `summary.json` result files.
- Include explicit `seed` and `hardware_label` fields.
- Separate hardware-specific results by output directory and `hardware_label`.
- Do not make claims about dynamic-loader latency improvement unless measurements show it.

## Execution Model

Act as the autonomous main agent for the whole implementation. Track overall progress in `qaq/doc/tasks/progress.md`, decompose work into modules, spawn subagents for independent modules when useful, integrate their results, and continue through implementation, tests, and validation without human-in-the-loop checkpoints unless genuinely blocked.

Subagents and worker threads are allowed, but their write scopes must be disjoint. Tell every worker that other agents may be editing nearby files, so they must inspect current file contents before modifying, must not revert unrelated edits, and must adapt to concurrent changes.

Keep changes scoped to `qaq/`, `requirements.txt` only if a test dependency is required, and small documentation updates that reflect implemented behavior. Do not edit unrelated research artifacts.

## Module Plan

### Workstream 1: Bit-Plane Weight Representation

Owner scope:

- `qaq/code/bitplanes.py`
- `qaq/tests/test_bitplanes.py`

Implement deterministic tensor-level quantization and reconstruction first, before model loading.

Required behavior:

- Define a `BitPlaneConfig` and `BitPlaneTensor` record using dataclasses or similarly explicit structured objects.
- Support initial `max_bits=8` and allowed widths including `2`, `4`, `6`, and `8`.
- Use symmetric per-tensor quantization unless the detailed design is explicitly updated.
- Convert source tensors to `float32` for quantization math.
- Guard zero tensors with scale `1.0`.
- Reject NaN or Inf values with clear errors that identify the tensor when available.
- Handle non-contiguous tensors without changing logical values.
- Split signed integer payloads into most-significant-first bit planes.
- Preserve signedness metadata so negative values reconstruct correctly.
- Implement `quantize_tensor_to_bitplanes(weight, config, tensor_name=None, group_id=None)`.
- Implement `reconstruct_tensor(record, bit_width, dtype=None, device=None)`.
- Implement `estimate_storage_bytes(record, bit_width=None)`.
- Exact max-bit reconstruction means exact reconstruction of the quantized/dequantized value, not exact recovery of the original float tensor.

Tests must cover signed and unsigned-like toy tensors, 2/4/6/8-bit reconstruction, exact max-bit quantized reconstruction, zero tensors, NaN/Inf rejection, non-contiguous tensors, shape preservation, unsupported bit-width rejection, and byte accounting.

### Workstream 2: Model Adapter and Inventory

Owner scope:

- `qaq/code/model_adapter.py`
- `qaq/tests/test_model_adapter.py`

Implement the shared inventory layer consumed by bit planes, policies, routing, and evaluation.

Required behavior:

- Define an inventory record with `tensor_name`, `module_name`, `module_role`, `layer_idx`, `group_id`, `group_granularity`, `shape`, `source_dtype`, and `target_quantized`.
- Discover selected Hugging Face causal-LM linear modules without mutating model weights.
- Support group granularities:
  - `transformer_layer`
  - `attention_mlp`
  - `linear_module`
- Classify LLaMA/Qwen-style module roles such as attention, MLP, embedding/lm head if selected, and a safe unknown fallback.
- Extract layer indices from common module paths such as `model.layers.12.self_attn.q_proj` and Qwen-style equivalents.
- Generate stable group ids that do not depend on raw traversal position alone.
- Serialize inventory records to JSON-compatible dictionaries.
- Fail clearly when no selected target modules are found.

Tests should use synthetic `torch.nn.Module` trees rather than target model downloads. Cover stable ids, role assignment, shape capture, group granularities, serialization, and empty-selection failure.

### Workstream 3: Static Mixed-Precision Policies

Owner scope:

- `qaq/code/policies.py`
- `qaq/tests/test_policies.py`

Implement fixed and hand-authored policies before router integration.

Required behavior:

- Define `PrecisionPolicy` and an expanded policy mapping contract.
- Implement built-ins:
  - `static_8bit`
  - `static_4bit`
  - `mixed_attention_high`
  - `random_router_baseline`
  - oracle-policy loading from JSON artifacts
- Expand every policy into a complete `group_id -> bit_width` mapping over the inventory.
- Validate unknown group ids, missing assignments without defaults, unsupported bit widths, and empty inventories.
- Save expanded mappings and policy metadata as JSON-compatible artifacts.
- Make `random_router_baseline` deterministic for a fixed seed.

Tests should use synthetic inventories. Cover fixed policies, attention/MLP mixed policies, seeded random determinism, invalid policy rejection, and JSON round trips.

### Workstream 4: Query Router and Oracle Labels

Owner scope:

- `qaq/code/router.py`
- `qaq/code/oracle_labels.py`
- `qaq/tests/test_router.py`

Implement the supervised router path with policy-id classification first. Do not jump directly to group-wise prediction or end-to-end soft routing until the supervised path is working.

Required behavior:

- Extract token-only features with `sample_id`, prompt metadata, input length, attention length, and `feature_schema_version`.
- Implement oracle-label selection from candidate policy scores using quality tolerance, expected cost, and a `tolerance_satisfied` flag.
- If no candidate satisfies the tolerance, emit the highest-precision or best-quality fallback with `tolerance_satisfied=false`; do not hide the failure.
- Train a lightweight deterministic policy-id classifier on features and oracle labels. Prefer a small PyTorch module or a transparent nearest/linear classifier that does not require adding scikit-learn.
- Predict policy ids, expand them through `policies.py`, and validate every selected group bit width.
- Save and load router artifacts with feature-schema compatibility checks.
- Emit router trace records with sample id, features, predicted or oracle policy, selected group bit widths, expected cost, quality metric when known, and confidence when available.

Tests must cover feature extraction, oracle selection including no-satisfying-candidate cases, training/prediction on tiny synthetic data, policy legality, artifact save/load, and schema mismatch rejection.

### Workstream 5: Dynamic Loader

Owner scope:

- `qaq/code/dynamic_loader.py`
- `qaq/tests/test_dynamic_loader.py`

Implement optional synchronous CPU-resident materialization after core reconstruction works.

Required behavior:

- Keep canonical bit-plane records CPU-resident.
- For each selected tensor, transfer or materialize only the planes needed by the selected bit width.
- Reconstruct through the shared `bitplanes.py` API.
- Measure `transfer_ms`, `reconstruction_ms`, `total_loader_ms`, `cpu_plane_bytes`, and `gpu_materialized_bytes`.
- Validate target device availability.
- Provide CPU-only dry-run behavior for tests.
- Keep loader metrics explicitly labeled and separate from non-loader latency summaries.

Tests must cover CPU-only materialization, target device validation, timing-field presence, and equality with non-loader reconstruction for the same bit width.

### Workstream 6: Evaluation Harness

Owner scope:

- `qaq/code/evaluate.py`
- `qaq/tests/test_evaluate.py`
- `qaq/results/` only for tiny generated smoke artifacts if needed

Implement the CLI coordinator after the lower-level modules have tests.

Required subcommands:

- `prepare-bitplanes`
- `run-static`
- `build-oracle-labels`
- `train-router`
- `run-router`
- `run-dynamic-loader`

Required behavior:

- Use `argparse`.
- Serialize the parsed run config into each output directory.
- Support synthetic examples or mocked/tiny smoke mode so CLI parsing, aggregation, and toy runs do not require target-model access.
- Support model name or local path for the target model and fail fast when unavailable outside explicit smoke mode.
- Wire model adapter, bit-plane store, policy expansion, router artifacts, and dynamic-loader paths through clear subcommands.
- Write per-sample `metrics.jsonl` with `sample_id`, `policy_name`, `seed`, `hardware_label`, token counts, quality, latency, memory, selected bit widths, and loader mode.
- Write `summary.json` with model, dataset, policies, quality, latency, memory, selected-bit distribution, dynamic-loader flag, errors, and hardware label.
- Label end-to-end timing honestly if prefill/decode timing is not implemented yet.
- Save optional `policy_trace.json`, `features.jsonl`, `oracle_labels.jsonl`, `candidate_scores.jsonl`, and router artifacts when the relevant mode produces them.

Tests must cover CLI parsing, config serialization, synthetic metrics aggregation, summary generation with and without loader fields, and one end-to-end toy or mocked-model run.

## Integration Order

1. Create the `qaq/code` and `qaq/tests` package skeleton.
2. Implement and test `bitplanes.py`.
3. Implement and test `model_adapter.py`.
4. Implement and test `policies.py`.
5. Implement and test `oracle_labels.py` and `router.py`.
6. Implement and test `dynamic_loader.py`.
7. Implement and test `evaluate.py` subcommands with synthetic and mocked paths.
8. Add a tiny smoke invocation documented in `qaq/README.md` or `qaq/doc` if useful.
9. Update `qaq/doc/tasks/progress.md` and each task file's checklist as work is completed.
10. Run all quality gates and fix failures.

## Testing and Quality Gates

Use repository facts, not assumed tooling. The current environment has `python3` but no global `pytest`, `ruff`, or `mypy`. If using pytest, add `pytest` to `requirements.txt` or otherwise document and provide a standard-library `unittest` alternative. Prefer pytest because the task files explicitly call for tests.

Required local gates before finishing:

```bash
python3 -m compileall qaq/code qaq/tests
python3 -m pytest qaq/tests
```

If you add or discover formatter/linter/type-checker configuration, also run the configured commands. Do not invent mandatory `ruff` or `mypy` gates unless you add and configure them intentionally.

Required functional validation:

- Tensor-only tests pass without model access.
- Synthetic inventory and policy tests pass without model access.
- Router and oracle-label tests pass without model access.
- Dynamic-loader tests pass on CPU-only environments.
- Evaluation harness aggregation tests pass without target-model access.
- At least one explicit smoke command runs successfully and writes valid `metrics.jsonl` and `summary.json` under `qaq/results/` or a temporary test output directory.

Target-model validation should be implemented as runnable commands but may be skipped in environments without gated model access or sufficient hardware. If skipped, record the reason in the final implementation notes.

## Acceptance Criteria

The implementation is complete when:

- `qaq/code/` contains working modules for bit planes, model inventory, policies, router/oracle labels, dynamic loading, and evaluation.
- `qaq/tests/` contains focused tests for every module listed in the task files.
- `qaq/doc/tasks/progress.md` accurately reflects completed modules.
- Bit-plane reconstruction works deterministically at selected bit widths and exact max-bit quantized reconstruction is tested.
- Policies expand to legal group mappings and serialize complete artifacts.
- The router can train and predict on synthetic labels, then emit legal per-query policies.
- The evaluation harness can run synthetic or mocked smoke modes and write repository-style `metrics.jsonl` and `summary.json`.
- Dynamic-loader metrics are present and separated when loader mode is enabled.
- `python3 -m compileall qaq/code qaq/tests` passes.
- `python3 -m pytest qaq/tests` passes, with `pytest` made available through requirements or an explicitly documented equivalent.
- Any target-model limitations, unresolved hardware/model access issues, and skipped target runs are clearly recorded.

## Uncertainty Protocol

Make conservative implementation choices when the docs leave room:

- Use symmetric per-tensor quantization first.
- Use policy-id classification before direct group-wise router prediction.
- Use token-only features as the first always-runnable router feature set.
- Use in-memory bit-plane storage first; add persistence only if it fits cleanly after tests pass.
- Use synthetic and mocked tests for CI-like coverage, and keep target-model runs as explicit hardware/model-access validations.

Known unresolved research choices from the docs:

- Final target-model access path: Hugging Face id, local checkpoint, or mirror.
- Primary hardware label for final evidence.
- First production comparison dataset and split.
- Final quality tolerance for oracle labels.
- Whether final grouping should be transformer layer, attention/MLP, or individual linear module.

Do not block implementation on these choices. Provide sensible CLI defaults, require explicit flags for target-model experiments, and document assumptions in result configs. Ask the user only if a missing choice makes forward progress impossible.
