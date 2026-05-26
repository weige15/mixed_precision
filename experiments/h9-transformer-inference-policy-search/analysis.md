# H9 Analysis

## 2026-05-26 Bootstrap

H9 starts after H8's selective-rescue close-out. The key pivot is from LoRA
fine-tuning to inference serving. The policy search is vLLM-first, model target
is `meta-llama/Llama-3.1-8B`, and the first hardware target is the lab RTX 3090.

The initial implementation should establish three things before any large
benchmark is trusted:

- which policy knobs are actually expressible by the installed vLLM runtime,
- which candidate policies instantiate successfully on the target GPU,
- which policies are non-dominated for prefill-heavy, decode-heavy, and mixed
  workloads.

The first scripts are infrastructure, not evidence yet. H9 should not report a
throughput, memory, or quality claim until a policy has a completed benchmark
artifact under a matched hardware label and workload definition.

## 2026-05-26 H9.1 RTX 3090 Results

The first H9.1 benchmark pass is complete for `meta-llama/Llama-3.1-8B` on the
lab RTX 3090 with vLLM. The completed full-workload policies are:

- `bf16_default`
- `fp16_default`
- `bf16_kv_fp8`
- `bf16_kv_fp8_e4m3`
- `fp16_kv_fp8`
- `fp16_kv_fp8_e4m3`
- `fp16_bitsandbytes`

`fp16_torchao` is not a valid H9.1 policy yet. vLLM requires an explicit
`torchao_config`, so the placeholder policy is intentionally skipped until a
concrete TorchAO configuration is added.

Quality scoring completed for all seven runnable policies. Each policy scored
1202 prompt tokens with zero missing logprobs. Prompt-NLL deltas versus
`bf16_default`:

| Policy | Mean prompt NLL | Delta vs bf16 |
|---|---:|---:|
| `bf16_default` | 0.613674 | 0.0000% |
| `fp16_default` | 0.613730 | +0.0090% |
| `bf16_kv_fp8` | 0.613373 | -0.0491% |
| `bf16_kv_fp8_e4m3` | 0.613373 | -0.0491% |
| `fp16_kv_fp8` | 0.612501 | -0.1912% |
| `fp16_kv_fp8_e4m3` | 0.612501 | -0.1912% |
| `fp16_bitsandbytes` | 0.621800 | +1.3241% |

The first six policies pass the 1% quality gate. `fp16_bitsandbytes` fails the
quality gate.

Performance and memory summary:

| Policy | Workload | Latency delta | Output-throughput delta | Memory delta | Quality delta |
|---|---|---:|---:|---:|---:|
| `fp16_default` | prefill-heavy | -0.53% | +0.54% | +0.00% | +0.0090% |
| `fp16_default` | mixed | -0.45% | +0.45% | +0.00% | +0.0090% |
| `fp16_default` | decode-heavy | +0.27% | -0.27% | +0.00% | +0.0090% |
| FP8 KV variants | all full workloads | roughly 0% to +2.5% latency | no throughput win | +3.64% to +3.65% | pass |
| `fp16_bitsandbytes` | decode-heavy | -42.35% | +73.46% | +3.00% | +1.3241% |
| `fp16_bitsandbytes` | mixed | +122.95% | -55.15% | +2.99% | +1.3241% |

Interpretation:

- `fp16_default` is the only clean H9.1 Pareto-safe candidate so far. It keeps
  quality and memory essentially unchanged while slightly improving prefill and
  mixed workloads.
- FP8 KV-cache policies pass the quality gate but are dominated in this setup:
  they do not reduce measured memory and they slightly slow the benchmark
  workloads.
- `fp16_bitsandbytes` is an interesting systems artifact, not a supported H9.1
  policy. It improves decode-heavy throughput substantially, but fails the 1%
  prompt-NLL gate and is much worse on the mixed workload. Its prefill-heavy
  run generated only one token instead of the requested 16, so that row should
  not be used as a normal prefill throughput comparison.

H9.1 is therefore partially supported. The infrastructure can search real vLLM
backend policies and identify dominated versus non-dominated candidates, but
the first policy grid does not find a strong memory-saving policy on the RTX
3090. The immediate publishable result is a careful systems finding: default
`fp16` is a safe small improvement over `bf16`; FP8 KV cache is not beneficial
under this vLLM/GPU/prompt setup; bitsandbytes needs either quality rescue,
different artifacts, or a narrower decode-only framing before it can be treated
as useful.

## 2026-05-26 H9.2 Setup

H9.2 targets the main unresolved question from H9.1: FP8 KV-cache policies may
need longer contexts before their intended memory benefit appears. The focused
H9.2 policy file is:

- `results/h9_2_long_context_policy_candidates.json`

It contains six policies:

- `bf16_default`
- `fp16_default`
- `bf16_kv_fp8_e4m3`
- `fp16_kv_fp8_e4m3`
- `bf16_kv_fp8`
- `fp16_kv_fp8`

It intentionally excludes `fp16_bitsandbytes`, because H9.2 is about KV-cache
stress rather than weight quantization, and H9.1 bitsandbytes failed the quality
gate. It also excludes `fp16_torchao`, which still lacks a concrete
`torchao_config`.

The H9.2 workloads are:

- `prefill_4k`: near-4k prompt, 16-token generation.
- `decode_2k_context`: long prompt, 128-token generation.
- `batch_mixed_long`: four long prompts, 64-token generation.

The dry-run benchmark plan was generated successfully under:

- `results/h9_2_long_context_benchmarks/`

No H9.2 runtime evidence exists yet. The next empirical step is to run the six
policies one process at a time on the RTX 3090, then summarize with
`h9_2_long_context_summary.json`.

## 2026-05-26 H9.2 Long-Context RTX 3090 Results

The H9.2 benchmark pass completed for all six focused policies on the lab RTX
3090:

- 6 policy artifacts
- 18 completed policy-workload rows
- 0 failed policies
- 3 repeats per policy-workload

The workloads are long-context stress tests rather than the shorter H9.1 suite:

- `prefill_4k`: 3542 prompt tokens, 16 generated tokens.
- `decode_2k_context`: long prompt, 128 generated tokens.
- `batch_mixed_long`: four long prompts, 64 generated tokens each.

The main result is negative for the original FP8 KV-cache memory hypothesis.
Even under longer-context pressure, every FP8 KV-cache policy used about
`22.678 GiB`, while the default KV-cache policies used about `21.879-21.881
GiB`. That is a measured memory increase of about `+3.6%`, not a memory saving.

H9.2 benchmark summary:

| Workload | Best/important policy | Latency delta vs `bf16_default` | Memory delta | Interpretation |
|---|---|---:|---:|---|
| `batch_mixed_long` | `fp16_kv_fp8` | -2.06% | +3.64% | Fastest batch-long policy, but memory-costly. |
| `batch_mixed_long` | `fp16_default` | -0.43% | +0.00% | Clean small win with no memory cost. |
| `decode_2k_context` | `bf16_default` | 0.00% | +0.00% | Best decode-context baseline; FP8 KV variants are slower and larger. |
| `prefill_4k` | `bf16_kv_fp8_e4m3` | -3.61% | +3.65% | Fastest prefill policy, but memory-costly. |
| `prefill_4k` | `fp16_default` | -2.06% | +0.00% | Clean prefill improvement with no memory cost. |

Interpretation:

- H9.2 does not rescue FP8 KV cache as a memory-saving policy on this
  RTX 3090/vLLM stack. It can produce small latency improvements in prefill and
  long-batch settings, but the measured memory direction is wrong.
- Decode remains the least favorable regime for FP8 KV cache. The
  `decode_2k_context` workload is best served by `bf16_default`; all KV-cache
  variants add memory and slightly increase latency.
- `fp16_default` remains the safest backend-real policy found so far. It keeps
  measured memory unchanged and improves `prefill_4k` latency by about `2.06%`
  and `batch_mixed_long` latency by about `0.43%`, with only the earlier H9.1
  small decode regression.
- The H9.2 summary currently links to the H9.1 quality directory. Therefore the
  quality deltas in `h9_2_long_context_summary.json` are policy-level prompt-NLL
  diagnostics from the earlier prompt suite, not a long-context-specific quality
  measurement. Before making a final H9.2 quality claim, run quality scoring
  with `--policies results/h9_2_long_context_policy_candidates.json` and a
  separate `results/h9_2_long_context_quality/` output directory.

H9.2 therefore strengthens the H9.1 conclusion: the infrastructure can evaluate
backend-real vLLM policies by workload, but the current vLLM-accessible FP8 KV
cache knob is not the missing memory-saving policy on the RTX 3090. The next
step toward a HAQ-style Transformer module assignment is not another global KV
dtype toggle; it is a richer search space over backend-supported module or
group policies, such as AWQ/GPTQ/Marlin artifacts, TorchAO configs, or
selective high-precision rescue from a quantized weight baseline.
