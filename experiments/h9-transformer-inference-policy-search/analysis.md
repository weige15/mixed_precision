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
