# H9 Protocol: Hardware-Aware Transformer Inference Policy Search

## Question

Can a hardware-aware policy search choose Transformer inference precision and backend policies that improve the latency-memory-quality Pareto frontier versus standard vLLM defaults?

## Motivation

H8 showed that a backend-real low-bit baseline plus selective high-precision rescue can improve the quality side of a memory-saving LoRA policy. H9 moves from fine-tuning to inference. The relevant hardware constraints change: prefill latency, decode throughput, KV-cache memory, model-weight format, attention backend, and vLLM-supported kernels matter more than optimizer stability.

HAQ is the template at the abstraction level: do not hand-pick one global bitwidth; search over backend-real policies under measured hardware cost. For Transformers, the policy space must separate at least:

- model weight dtype or quantization backend,
- KV-cache dtype,
- prefill-heavy versus decode-heavy workloads,
- runtime mode and attention/kernel implementation exposed by the serving stack.

## Hypothesis

A vLLM-backed policy search over model dtype, quantization backend, KV-cache dtype, and runtime knobs can identify non-dominated policies that improve memory or latency versus bf16/fp16 defaults while staying inside a small quality-degradation gate.

## Scope

First target:

```text
runtime: vLLM
model: meta-llama/Llama-3.1-8B
hardware: lab RTX 3090
budget: thorough single-GPU pilot
objective: Pareto frontier, not one scalar score
```

H9.1 does not implement custom kernels or fake quantization. A policy can only make a performance claim if vLLM actually instantiates the corresponding backend.

## Policy Space

Each policy is a concrete vLLM launch configuration:

```text
policy = {
  model,
  dtype,
  quantization,
  kv_cache_dtype,
  enforce_eager,
  block_size,
  gpu_memory_utilization,
  max_model_len
}
```

Initial candidates:

1. `bf16_default`: bf16 model dtype, default KV cache.
2. `fp16_default`: fp16 model dtype, default KV cache.
3. bf16/fp16 with FP8 KV cache if supported by the installed vLLM/GPU path.
4. bitsandbytes-backed weight quantization if vLLM can load the model with that backend.
5. torchao-backed weight quantization only if local package compatibility is sufficient.
6. eager-mode controls for bf16/fp16, used to diagnose compile/runtime effects rather than as expected winners.

AWQ, GPTQ, Marlin, and BitBLAS policies are H9.2 candidates unless compatible quantized model artifacts are already available.

## Workloads

H9 must benchmark at least three fixed workloads:

```text
prefill_heavy: long prompts, short generation
decode_heavy: short prompts, long generation
mixed: varied prompt lengths and generation lengths
```

This is mandatory because prefill and decode stress different parts of Transformer inference. A policy that wins prefill can lose decode.

## Metrics

Primary measured metrics:

- end-to-end latency per request batch,
- prefill-heavy latency proxy,
- decode-heavy output tokens/sec,
- total generated tokens/sec,
- peak CUDA allocated/reserved memory,
- load-time memory and post-run memory,
- policy failure reason if vLLM cannot instantiate the backend.

Quality metrics:

- preferred: held-out prompt negative log likelihood or perplexity using vLLM logprobs if feasible,
- fallback: deterministic output/logprob agreement against the bf16 baseline on a fixed prompt suite.

The default quality gate is `<= 1%` degradation versus bf16 if an NLL/perplexity metric is available. If H9.1 only has output/logprob agreement, quality is reported as diagnostic and the policy is not called quality-supported until the stronger metric is added.

## Search Procedure

1. Run backend inventory without loading the full model.
2. Generate a fixed candidate policy grid.
3. Run smoke benchmarks for each candidate on one tiny prompt.
4. Drop unsupported policies only after recording their failure status.
5. Run the full workload suite for supported candidates.
6. Summarize non-dominated policies separately for each workload and for aggregate means.

## Decision Rules

H9 is supported if at least one backend-real policy is non-dominated against bf16/fp16 defaults and improves either memory or latency while passing the quality gate.

H9 is partially supported if the search infrastructure works and reveals clear backend constraints, but no candidate passes the quality/performance trade-off.

H9 is not supported if candidate policies cannot be instantiated in vLLM on the target hardware, or if all viable policies are dominated by default bf16/fp16.

## Reporting Constraints

- Compare only runs with the same model, vLLM version, hardware label, prompt suite, and workload definition.
- Do not treat fake quantization or unsupported backend flags as performance evidence.
- Report prefill and decode separately before giving any aggregate conclusion.
- Record failed policies; they are systems evidence.
