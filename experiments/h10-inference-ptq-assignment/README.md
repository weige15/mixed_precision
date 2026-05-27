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

Older PEFT/QLoRA selective-rescue work is archived under:

```text
experiments/h10-haq-peft-assignment/
experiments/h10-peft-precision-risk/
```

Those artifacts may still support the sensitivity-probe story, but final H10
claims should use inference workloads, backend-real PTQ actions, and deployment
metrics such as prompt NLL, prefill latency, decode throughput, peak memory, and
KV-cache memory.
