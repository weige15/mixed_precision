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
