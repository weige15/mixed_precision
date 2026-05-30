# Proposal: QAQ Query-Adaptive Mixed-Precision Quantization

## Objective
Build a project-local implementation of QAQ, a query-adaptive mixed-precision
LLM inference prototype inspired by "QAQ: Query-adaptive Mixed-precision
Quantization for Large Language Models." The assignment goal is to explore
whether a single stored model can choose different precision levels per input
query, instead of serving every request with one fixed quantization policy.

The implementation should begin as a research prototype rather than a production
kernel project. It should make the algorithm measurable on modest hardware:
correctness, router behavior, memory footprint, latency, and quality should all
be reported against fixed-precision baselines.

Paper sources checked:

- OpenReview forum: https://openreview.net/forum?id=dpHfDasG44
- OpenReview PDF: https://openreview.net/pdf?id=dpHfDasG44
- NeurIPS virtual page: https://neurips.cc/virtual/2025/129098

## Current Project State
The repository is an existing mixed-precision LLM research workspace, not an
empty greenfield project. It currently uses `requirements.txt` with PyTorch,
Transformers, Datasets, PEFT, Accelerate, NumPy, Pandas, and tqdm. I did not
observe a `pyproject.toml` in the file discovery pass, so this proposal treats
`qaq/` as a new subproject inside the current requirements-based repository
unless the project is later converted to uv.

Relevant existing context:

- `README.md` frames the project around selective and adaptive precision for
  small LLM LoRA experiments.
- `literature/qaq_query_adaptive_quantization_2025.md` already summarizes QAQ
  as query-conditioned bit-plane precision selection with optional CPU/GPU
  loading.
- `experiments/h6-adaptive-precision-assignment/`,
  `experiments/h7-precision-predictor/`, and
  `experiments/h10-inference-ptq-assignment/` contain useful precedents for
  calibration, precision prediction, policy search, and hardware-aware
  evaluation.
- `qaq_deck/` exists as presentation material, but no `qaq/` programming
  directory existed before this proposal.

## Assumptions
This proposal makes the following explicit assumptions:

- The first deliverable should be a readable, testable Python prototype under
  `qaq/`, not a custom CUDA kernel.
- The initial model target should be a small Hugging Face causal LM that can run
  locally, with Qwen or LLaMA-family experiments deferred until hardware and
  access are confirmed.
- Weight quantization can start with fake or packed integer bit-plane
  reconstruction in PyTorch. Kernel-level speedups are a later milestone.
- The router should first select precision at transformer block or linear-module
  group granularity. Per-weight or per-channel routing is out of scope for the
  first version.
- On-demand CPU-to-GPU loading should be implemented after the bit-plane and
  router path is working, because transfer orchestration can obscure algorithmic
  correctness.

## Proposed Approach
Implement QAQ in four layers: bit-plane storage, precision reconstruction,
query router, and evaluation harness.

### 1. Bit-plane weight representation
Create utilities that quantize each selected weight tensor to a maximum bit
width, initially 8 bits, and split the quantized integer representation into
ordered bit planes. Store tensor metadata alongside the planes:

- original tensor name and shape
- quantization scale and zero point or symmetric scale
- maximum bit width
- block or module group assignment
- device placement and storage dtype

The first implementation should reconstruct approximate weights from the top-k
most significant bit planes for candidate bit widths such as 2, 4, 6, and 8.
This mirrors the QAQ paper's core storage idea while keeping the implementation
inspectable.

### 2. Static mixed-precision baseline
Before training a router, support fixed policies:

- all selected modules at 8-bit reconstruction
- all selected modules at 4-bit reconstruction
- simple hand-authored mixed policies, for example higher precision in attention
  and lower precision in MLP blocks

These baselines are required so query-adaptive routing can be compared against a
known static memory-quality-latency trade-off.

### 3. Query-conditioned router
Add a lightweight router that predicts a bit-width choice for each block or
module group from query features. Start with simple features that are easy to
compute and reproduce:

- input length
- pooled embedding or first-token hidden state from an early model pass
- activation outlier summaries from selected blocks, if available
- optional task or prompt metadata supplied by the evaluation harness

The router can be trained with a distillation-style objective:

- teacher: full precision or 8-bit reconstructed model outputs
- student: router-selected mixed-precision model outputs
- quality loss: KL divergence or next-token cross entropy against teacher logits
- cost loss: expected bit-width or estimated memory/latency penalty

For early experiments, include an oracle or offline label builder that evaluates
candidate bit-width policies per calibration sample and labels the cheapest
policy that stays within a quality tolerance. This gives the router a simpler
supervised target before moving to end-to-end soft routing.

### 4. Dynamic loader
Once reconstruction and routing are stable, add an optional loading layer that
keeps all bit planes in CPU memory and materializes only the selected planes on
GPU for each query. The first version can be synchronous and explicitly measured.
The proposal should not claim latency improvement from this path until the data
shows it; the QAQ paper reports memory reduction with latency overhead when
loading is synchronous.

### 5. Evaluation harness
Add scripts that run small, repeatable comparisons:

- quality: perplexity or next-token loss on WikiText-style text and a small
  instruction dataset subset
- memory: peak GPU memory and resident CPU storage
- latency: prefill and decode timing where possible, or end-to-end generation
  timing for the prototype
- routing behavior: distribution of selected bit widths by query, layer, and
  prompt length
- ablations: no router, random router, static 4-bit, static 8-bit, and oracle
  labels

The results should be written as JSONL plus a summary JSON, matching the style
used in the existing experiment directories.

## Milestones
1. Create the `qaq/` prototype skeleton with bit-plane decomposition,
   reconstruction utilities, and unit tests on toy tensors.
2. Wrap selected Hugging Face linear modules with reconstructed weights and
   reproduce static 4-bit and 8-bit inference baselines on a small model.
3. Build calibration scripts that score candidate bit-width policies per sample
   and produce supervised router labels.
4. Train and evaluate a lightweight query router against static baselines,
   reporting quality, latency, memory, and selected-bit distributions.
5. Add optional CPU-to-GPU on-demand loading and measure the memory-latency
   trade-off separately from the core algorithm.

## Open Questions
- Should `qaq/` become a uv-managed standalone Python project with its own
  `pyproject.toml`, or should it reuse the repository-level `requirements.txt`?
- Which first target model should be used for the assignment: a tiny smoke model,
  Qwen2.5-0.5B, Qwen3, or LLaMA-3.1?
- Which hardware should the implementation optimize for first: CPU-only smoke
  tests, a local RTX 3090-style GPU, or an A100-class environment?
- Should the first router operate at transformer-layer granularity, attention vs
  MLP block granularity, or individual linear-module granularity?
- What quality tolerance should define "minimal sufficient precision" during
  oracle label generation?

## Validation Plan
Validation should progress from deterministic unit tests to model-level
experiments:

1. Unit-test bit-plane split and reconstruction on signed and unsigned toy
   tensors, including exact reconstruction at max bit width.
2. Verify that static reconstructed weights produce deterministic logits for a
   fixed prompt and seed.
3. Compare static 4-bit, static 8-bit, and mixed policies on a small calibration
   set, recording loss, latency, and memory.
4. Train the router on one calibration split and evaluate on a held-out split.
5. Confirm that query-adaptive routing beats at least one static baseline on the
   assignment metric, for example similar quality to static 8-bit with lower
   average selected bits, or similar memory to static 4-bit with better quality.
6. If on-demand loading is enabled, report it separately with both memory savings
   and latency overhead so the result remains scientifically honest.
