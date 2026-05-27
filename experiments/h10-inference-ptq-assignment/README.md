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

Older PEFT/QLoRA selective-rescue work is archived under:

```text
experiments/h10-haq-peft-assignment/
experiments/h10-peft-precision-risk/
```

Those artifacts may still support the sensitivity-probe story, but final H10
claims should use inference workloads, backend-real PTQ actions, and deployment
metrics such as prompt NLL, prefill latency, decode throughput, peak memory, and
KV-cache memory.
