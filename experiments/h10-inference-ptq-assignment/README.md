# H10: Inference-Side Mixed-Precision PTQ Assignment

H10 is now scoped to HAQ-style mixed-precision post-training quantization for
LLM inference.

The active question is:

> Can short calibration and perturbation probes, combined with real vLLM backend
> measurements, choose an implementable mixed-precision inference policy that
> improves the latency-memory-quality trade-off versus bf16/fp16 defaults?

Start with:

```text
protocol.md
analysis.md
```

Current executable artifacts:

```bash
python experiments/h9-transformer-inference-policy-search/code/summarize_h9_results.py \
  --results-dir experiments/h9-transformer-inference-policy-search/results/benchmarks \
  --quality-dir experiments/h9-transformer-inference-policy-search/results/quality \
  --output experiments/h9-transformer-inference-policy-search/results/h9_benchmark_summary.json

python experiments/h9-transformer-inference-policy-search/code/summarize_h9_results.py \
  --results-dir experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_benchmarks \
  --quality-dir experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_quality \
  --output experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_summary.json

python experiments/h10-inference-ptq-assignment/code/build_inference_action_table.py
python experiments/h10-inference-ptq-assignment/code/solve_inference_assignment.py
```

Final matched Instruct Marlin artifacts:

```text
results/action_table_final_instruct_marlin.csv
results/selected_policy_final_instruct_marlin_strict.json
results/solver_trace_final_instruct_marlin_strict.json
results/selected_policy_final_instruct_marlin_relaxed_3pct.json
results/solver_trace_final_instruct_marlin_relaxed_3pct.json
```

The strict final result selects
`llama31_8b_instruct_gptq_marlin_artifact` under the 1% prompt-NLL gate. The
matched GPTQ-Marlin rows improve latency by about 60-63% and output throughput
by about 153-168%, with a +0.773771% prompt-NLL delta versus matched
`bf16_default`.

To add new backend-real PTQ evidence, first run the configured H9 TorchAO
candidates on a CUDA host:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_benchmark.py \
  --policy-name fp16_torchao_int8wo \
  --policy-name fp16_torchao_int8dyn_int8w \
  --policy-name fp16_torchao_int4wo_g128 \
  --smoke \
  --repeats 1 \
  --warmup-runs 0 \
  --hardware-label rtx3090-lab
```

Only after smoke success, run full benchmark and quality passes for the
surviving TorchAO policies, regenerate H9 summaries, then rerun the H10 build
and solve commands above.

The preferred next path after the TorchAO loader failure is artifact-backed
PTQ. Add a local AWQ/GPTQ/Marlin artifact policy to H9, smoke-test it, then
let H10 ingest the resulting H9 benchmark and quality artifacts:

```bash
python experiments/h9-transformer-inference-policy-search/code/generate_h9_policies.py \
  --artifact-policies experiments/h9-transformer-inference-policy-search/artifact_policies.local.json \
  --output experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates.json

CUDA_VISIBLE_DEVICES=0 \
python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_benchmark.py \
  --policy-name llama31_8b_awq_artifact \
  --smoke \
  --repeats 1 \
  --warmup-runs 0 \
  --hardware-label rtx3090-lab
```

The H10 table and solver outputs are:

```text
results/action_table.csv
results/selected_policy.json
results/solver_trace.json
```

For final Instruct artifact claims, use the matched Marlin table instead:

```bash
python experiments/h10-inference-ptq-assignment/code/build_inference_action_table.py \
  --skip-default-summaries \
  --extra-h9-summary h9_instruct_awq_marlin=experiments/h9-transformer-inference-policy-search/results/h9_instruct_awq_marlin_summary.json \
  --extra-policy-candidates experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates_instruct_awq_marlin.json \
  --extra-h9-summary h9_instruct_gptq_marlin=experiments/h9-transformer-inference-policy-search/results/h9_instruct_gptq_marlin_summary.json \
  --extra-policy-candidates experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates_instruct_gptq_marlin.json \
  --output experiments/h10-inference-ptq-assignment/results/action_table_final_instruct_marlin.csv

python experiments/h10-inference-ptq-assignment/code/solve_inference_assignment.py \
  --action-table experiments/h10-inference-ptq-assignment/results/action_table_final_instruct_marlin.csv \
  --output experiments/h10-inference-ptq-assignment/results/selected_policy_final_instruct_marlin_strict.json \
  --trace-output experiments/h10-inference-ptq-assignment/results/solver_trace_final_instruct_marlin_strict.json \
  --quality-epsilon 0.01

python experiments/h10-inference-ptq-assignment/code/solve_inference_assignment.py \
  --action-table experiments/h10-inference-ptq-assignment/results/action_table_final_instruct_marlin.csv \
  --output experiments/h10-inference-ptq-assignment/results/selected_policy_final_instruct_marlin_relaxed_3pct.json \
  --trace-output experiments/h10-inference-ptq-assignment/results/solver_trace_final_instruct_marlin_relaxed_3pct.json \
  --quality-epsilon 0.03
```

Next task-quality check for the strict H10 claim:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/run_h10_task_quality.py \
  --policies experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates_instruct_gptq_marlin.json \
  --policy-name bf16_default \
  --policy-name fp16_default \
  --policy-name llama31_8b_instruct_gptq_marlin_artifact \
  --hardware-label rtx3090-lab
```

This complements prompt-NLL scoring with a small deterministic exact-match
screen. Treat it as a regression check, not a replacement for a full downstream
benchmark suite.

If the RTX 3090 host is occupied, an A100 run is acceptable as a separate
hardware stratum. Do not mix A100 latency or memory numbers with the RTX 3090
claim. For A100, write into a separate run label and compare GPTQ-Marlin only
against A100 bf16/fp16 outputs:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/run_h10_task_quality.py \
  --policies experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates_instruct_gptq_marlin.json \
  --policy-name bf16_default \
  --policy-name fp16_default \
  --policy-name llama31_8b_instruct_gptq_marlin_artifact \
  --hardware-label a100-lab \
  --run-label a100-lab
```

If task quality passes on A100, it is useful robustness evidence for the
selected policy. It is not a replacement for the matched RTX 3090 deployment
frontier unless the H9 benchmark and prompt-NLL summaries are also rerun on
A100 and stored as A100-specific artifacts.

On Colab, vLLM wheels may be incompatible with the installed CUDA or
Transformers stack. For task quality only, use the Transformers backend and
override the lab-local GPTQ path with the Hugging Face artifact id:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/run_h10_task_quality.py \
  --runtime-backend transformers \
  --policies experiments/h9-transformer-inference-policy-search/results/h9_policy_candidates_instruct_gptq_marlin.json \
  --policy-name bf16_default \
  --policy-name fp16_default \
  --policy-name llama31_8b_instruct_gptq_marlin_artifact \
  --model-override llama31_8b_instruct_gptq_marlin_artifact=shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ \
  --hardware-label a100-colab \
  --run-label a100-colab-transformers
```

The Transformers backend is for functional task-quality robustness checks. Do
not use it for the vLLM deployment-frontier latency or memory claim.

## One-Command Artifact PTQ Path

You do not need to train a model to test artifact-backed PTQ. Use an existing
quantized checkpoint from Hugging Face or a local path that vLLM can load.

## Layer/Group Backend Path

The first true layer/group backend path is now implemented with Transformers
plus TorchAO `FqnToConfig`. It is separate from the vLLM artifact path because
vLLM does not currently expose arbitrary per-layer precision overrides for a
normal Hugging Face checkpoint in this project setup.

Generate the Llama layer/group policies:

```bash
python experiments/h10-inference-ptq-assignment/code/generate_layer_group_policies.py
```

The generated grid includes matched `bf16_transformers` and `fp16_transformers`
baselines plus starter TorchAO policies that quantize late Llama MLP projection
groups through regex FQN matching.

Validate wiring without loading the model:

```bash
python experiments/h10-inference-ptq-assignment/code/run_layer_group_backend.py \
  --policy-name h10_lg_late_gate_up_int8wo \
  --dry-run \
  --repeats 1 \
  --warmup-runs 0
```

Run the CUDA benchmark and prompt-NLL quality pass:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/run_layer_group_backend.py \
  --policy-name bf16_transformers \
  --policy-name fp16_transformers \
  --policy-name h10_lg_late_gate_up_int8wo \
  --policy-name h10_lg_late_mlp_int8wo \
  --hardware-label rtx3090-lab
```

Summarize and solve with the matched Transformers baseline:

```bash
python experiments/h9-transformer-inference-policy-search/code/summarize_h9_results.py \
  --results-dir experiments/h10-inference-ptq-assignment/results/layer_group_benchmarks \
  --quality-dir experiments/h10-inference-ptq-assignment/results/layer_group_quality \
  --baseline-policy bf16_transformers \
  --output experiments/h10-inference-ptq-assignment/results/layer_group_summary.json

python experiments/h10-inference-ptq-assignment/code/build_inference_action_table.py \
  --skip-default-summaries \
  --policy-candidates experiments/h10-inference-ptq-assignment/results/layer_group_policy_candidates.json \
  --extra-h9-summary h10_layer_group=experiments/h10-inference-ptq-assignment/results/layer_group_summary.json \
  --output experiments/h10-inference-ptq-assignment/results/action_table_layer_group.csv

python experiments/h10-inference-ptq-assignment/code/solve_inference_assignment.py \
  --action-table experiments/h10-inference-ptq-assignment/results/action_table_layer_group.csv \
  --baseline-policy bf16_transformers \
  --output experiments/h10-inference-ptq-assignment/results/selected_policy_layer_group.json \
  --trace-output experiments/h10-inference-ptq-assignment/results/solver_trace_layer_group.json
```

This path is backend-real once run on CUDA: the model is loaded with
Transformers, selected Llama module FQNs are quantized in place with TorchAO,
and the benchmark/quality artifacts use the same schema as H9 summaries. Until
those CUDA artifacts complete, it should be treated as implementation readiness,
not empirical H10 support.

List built-in downloadable candidates:

```bash
python experiments/h10-inference-ptq-assignment/code/download_quantized_artifact.py \
  --list-candidates
```

Download the default base-model GPTQ candidate:

```bash
python experiments/h10-inference-ptq-assignment/code/download_quantized_artifact.py
```

Download and immediately run the CUDA smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/download_quantized_artifact.py \
  --run-smoke \
  --hardware-label rtx3090-lab
```

Download and run the full benchmark, quality scoring, H9 summary, H10
action-table build, and H10 solver:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/download_quantized_artifact.py \
  --run-full \
  --hardware-label rtx3090-lab
```

If the quality step fails with a TorchDynamo/TorchInductor C++ compile error
ending in `No space left on device` for `/tmp`, rerun only the failed quality
and summary stages with compiler temp files redirected into the repo:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/download_quantized_artifact.py \
  --candidate llama31_8b_base_gptq \
  --run-full \
  --skip-smoke \
  --skip-benchmark \
  --runtime-cache-dir tmp/h10_ptq_runtime_cache \
  --hardware-label rtx3090-lab
```

If the benchmark did not complete either, omit `--skip-benchmark`.

Default smoke test, using `shuyuej/Meta-Llama-3.1-8B-GPTQ`:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/run_artifact_ptq_pipeline.py \
  --smoke-only \
  --hardware-label rtx3090-lab
```

If the smoke test completes, run the full benchmark, quality scoring, H9
summary, H10 action-table build, and H10 solver:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/run_artifact_ptq_pipeline.py \
  --hardware-label rtx3090-lab
```

For a different quantized artifact:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-inference-ptq-assignment/code/run_artifact_ptq_pipeline.py \
  --policy-name llama31_8b_awq_artifact \
  --model-name hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --hardware-label rtx3090-lab
```

Only compare against existing H9 base-model baselines when the artifact uses the
same base model. If the artifact is Llama-3.1-8B-Instruct, regenerate matching
bf16/fp16 baselines for the Instruct model before making quality claims.

Older PEFT/QLoRA selective-rescue work is archived under:

```text
experiments/h10-haq-peft-assignment/
experiments/h10-peft-precision-risk/
```

Those artifacts may still support the sensitivity-probe story, but final H10
claims should use inference workloads, backend-real PTQ actions, and deployment
metrics such as prompt NLL, prefill latency, decode throughput, peak memory, and
KV-cache memory.
