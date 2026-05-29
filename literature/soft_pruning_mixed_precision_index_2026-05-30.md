# Focused Index: Soft Pruning as Adaptive Mixed-Precision Information Allocation

## Framing

The most useful way to state the idea is not "pruning" in the ordinary binary
sense. It is information allocation:

- 0 bits: remove or ignore the component.
- Few bits: preserve coarse information.
- More bits: preserve sensitive or prompt-critical information.
- Extra residual bits/codewords/bit planes: optional refinement.

This turns soft pruning into a learnable or searchable bitwidth decision. The
open problem is making those decisions deployable: heterogeneous bitwidths need
packing, metadata, aligned memory access, and kernels that can consume the final
layout.

## Paper Map

| Category | Best papers | What they contribute |
|---|---|---|
| Soft bit allocation | Bayesian Bits, DiffQ, FracBits, DNAS mixed precision | Differentiable or relaxed bitwidth search before final hard assignment. |
| Importance/salience allocation | SliM-LLM, CMPQ, AWQ, OWQ, SPQR, HAWQ | Higher precision is reserved for sensitive weights, channels, groups, or outliers. |
| Prompt/query-conditioned precision | QAQ, Prompt-Adaptive Quantization, Instance-Aware Dynamic Quantization | The input can decide how much precision is needed. |
| Hardware-aware assignment | HAQ, HAWQ-V3, ScaleBITS, RAMP, FGMP, FineQ | Bitwidth policy is constrained by memory, latency, energy, block layout, and kernel support. |
| Storage-efficient multi-precision | Matryoshka Quantization, AdaQuantLM, QAQ, MoBiQuant | Avoid keeping many full model copies by nesting bit planes, residual slices, or additive codewords. |

## Most Relevant Papers For The Proposed Direction

1. **Bayesian Bits**: best conceptual bridge from soft pruning to mixed precision;
   0-bit means pruning, higher bits mean more retained information.
2. **Matryoshka Quantization**: best storage-efficient "single model, multiple
   precisions" reference.
3. **QAQ**: best query/prompt-conditioned bit-plane reference, including
   CPU/GPU loading trade-offs.
4. **SliM-LLM**: best salience-driven group-wise LLM bit allocation reference.
5. **ScaleBITS**: best recent hardware-aligned bitwidth allocation reference.
6. **FineQ / FGMP**: best systems references for packing, alignment, and
   hardware support for fine-grained mixed precision.
7. **Prompt-Adaptive Quantization**: useful simple baseline that routes prompts
   among separate pre-quantized checkpoints.
8. **AdaQuantLM**: useful additive-codeword alternative to separate checkpoints.
9. **MoBiQuant**: highly related emerging token-adaptive residual-bit-slice
   method; use cautiously because no code or peer-review signal was found yet.

For a stricter venue/review-quality filter, use
[Review-Filtered Selection: Soft Pruning / Adaptive Mixed Precision](review_filtered_soft_pruning_mixed_precision_2026-05-30.md).

For the MoBiQuant-centered survey, use
[MoBiQuant-Like Papers For Soft-Pruning-Style Mixed Precision](mobiquant_like_soft_pruning_survey_2026-05-30.md).

## Fit To The User's Idea

### Importance from prompt

QAQ, MoBiQuant, and Prompt-Adaptive Quantization support the claim that easy and
hard inputs should not necessarily use the same precision. QAQ and MoBiQuant are
more aligned with the storage goal because they decompose weights into bit
planes or residual bit slices; PAQ is easier to explain but stores or serves
multiple quantized variants.

### Importance from hardware

HAQ, ScaleBITS, FineQ, and FGMP show that bit allocation must be hardware-aware.
The same sensitivity policy can be useful or useless depending on whether the
layout maps to aligned loads, supported kernels, and realistic memory movement.

### Importance from both

The most promising formulation is a two-factor policy:

```text
precision = f(component sensitivity, prompt difficulty, hardware cost)
```

Where component sensitivity can come from calibration signals, prompt difficulty
can come from a small router or activation statistics, and hardware cost comes
from a measured backend table.

## Storage Difficulty

Heterogeneous bitwidths create four practical problems:

1. **Packing:** arbitrary per-weight bitwidths fragment memory.
2. **Metadata:** indices, masks, scales, and bitwidth maps can erase savings.
3. **Alignment:** GPUs and accelerators prefer regular blocks, not scattered
   variable-length values.
4. **Kernel support:** fake mixed precision can preserve quality but produce no
   speed or memory win unless kernels consume the layout directly.

The literature's main answers are:

- Use nested bit planes or residual bit slices: Matryoshka Quantization, QAQ,
  MoBiQuant.
- Use additive codewords: AdaQuantLM.
- Use structured groups/blocks/channels: SliM-LLM, CMPQ, ScaleBITS.
- Protect outliers with compact encodings: FineQ, SPQR, OWQ.
- Co-design the hardware datapath: FGMP, FineQ.

## Research Hypothesis To Test Next

Disentangle model information into a base representation plus optional residual
bit planes:

```text
weight = essential_component + refinement_component
```

The essential component should preserve common prompt behavior and stay resident
in low-bit memory. Refinement components should be loaded or activated only for
high-risk modules, difficult prompts, or hardware states where extra precision is
worth the cost.

## Suggested Evaluation

- Quality: prompt NLL, task accuracy, and stress prompts routed to high precision.
- Compression: effective bits per parameter including metadata.
- Runtime: prefill latency, decode latency, memory bandwidth, GPU memory, and
  CPU/GPU transfer overhead if using on-demand bit planes.
- Routing validity: does the prompt router predict when low-bit inference fails?
- Layout validity: does the representation map to existing kernels or require a
  custom kernel?
