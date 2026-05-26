# H9: Transformer Inference Policy Search

H9 studies hardware-aware precision/backend policy search for Transformer inference.
The first target is `meta-llama/Llama-3.1-8B` in vLLM on the lab RTX 3090.

Main files:

- `protocol.md`: locked H9 question, hypothesis, metrics, and decision rules.
- `analysis.md`: running synthesis.
- `code/generate_h9_policies.py`: creates the first vLLM policy grid.
- `code/inspect_h9_backend_inventory.py`: records package/GPU/vLLM support and preflight policy status.
- `code/run_h9_vllm_benchmark.py`: runs one or more concrete vLLM policies on fixed workloads.
- `code/run_h9_vllm_quality.py`: runs prompt-logprob quality checks when supported.
- `code/summarize_h9_results.py`: aggregates benchmark outputs and marks Pareto candidates.

Generate the initial policy grid:

```bash
python experiments/h9-transformer-inference-policy-search/code/generate_h9_policies.py
```

Inspect local backend support without loading the full model:

```bash
python experiments/h9-transformer-inference-policy-search/code/inspect_h9_backend_inventory.py
```

Run a tiny smoke benchmark for one policy:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_benchmark.py \
  --policy-name bf16_default \
  --smoke \
  --hardware-label rtx3090-lab
```

Run a full benchmark for one policy:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_benchmark.py \
  --policy-name bf16_default \
  --repeats 3 \
  --hardware-label rtx3090-lab
```

Run prompt-logprob quality scoring for one policy:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_quality.py \
  --policy-name bf16_default \
  --hardware-label rtx3090-lab
```

Summarize completed policy runs:

```bash
python experiments/h9-transformer-inference-policy-search/code/summarize_h9_results.py
```

Use one process per policy for final measurements. vLLM allocates and caches GPU
memory aggressively, so separate invocations are cleaner than benchmarking every
policy in one Python process.

`fp16_torchao` is a placeholder until a concrete vLLM `torchao_config` is
added. Do not treat its config-missing failure as a backend performance result.

## H9.2 Long-Context KV Stress

Generate the focused H9.2 policy/workload file:

```bash
python experiments/h9-transformer-inference-policy-search/code/generate_h9_policies.py \
  --profile h9_2_long_context \
  --output experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_policy_candidates.json
```

Dry-run the long-context benchmark plan:

```bash
python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_benchmark.py \
  --policies experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_policy_candidates.json \
  --dry-run \
  --hardware-label rtx3090-lab
```

Run H9.2 one policy per process:

```bash
for p in bf16_default fp16_default bf16_kv_fp8_e4m3 fp16_kv_fp8_e4m3 bf16_kv_fp8 fp16_kv_fp8; do
  CUDA_VISIBLE_DEVICES=0 python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_benchmark.py \
    --policies experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_policy_candidates.json \
    --output-dir experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_benchmarks \
    --policy-name "$p" \
    --repeats 3 \
    --hardware-label rtx3090-lab
done
```

Summarize H9.2 separately from H9.1:

```bash
python experiments/h9-transformer-inference-policy-search/code/summarize_h9_results.py \
  --results-dir experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_benchmarks \
  --output experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_summary.json
```

Run long-context-specific prompt-logprob quality scoring for H9.2:

```bash
for p in bf16_default fp16_default bf16_kv_fp8_e4m3 fp16_kv_fp8_e4m3 bf16_kv_fp8 fp16_kv_fp8; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python experiments/h9-transformer-inference-policy-search/code/run_h9_vllm_quality.py \
    --policies experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_policy_candidates.json \
    --output-dir experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_quality \
    --policy-name "$p" \
    --quality-batch-size 1 \
    --gpu-memory-utilization 0.78 \
    --hardware-label rtx3090-lab
done
```

Regenerate the H9.2 summary with the long-context quality artifacts:

```bash
python experiments/h9-transformer-inference-policy-search/code/summarize_h9_results.py \
  --results-dir experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_benchmarks \
  --quality-dir experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_quality \
  --output experiments/h9-transformer-inference-policy-search/results/h9_2_long_context_summary.json
```
