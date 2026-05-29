# MoBiQuant-Like Papers For Soft-Pruning-Style Mixed Precision

## Selection Criterion

MoBiQuant is highly aligned with the current topic because it combines:

- token/prompt-dependent importance,
- variable bitwidth as an information budget,
- residual or nested storage so multiple precisions do not require multiple
  checkpoints,
- runtime routing under resource constraints.

This survey collects papers that share at least two of those properties. The
best matches are not ordinary uniform quantization papers; they are papers where
precision behaves like a soft keep/coarsen/drop decision.

## Closest Matches

| Rank | Paper | Why it is MoBiQuant-like | Code / review signal |
|---|---|---|---|
| 1 | [MoBiQuant](mobiquant_2026.md) | Token-aware router selects residual bit slices; single elastic mixed-bit representation. | arXiv 2026; no code found. |
| 2 | [DP-LLM](dp_llm_2025.md) | Runtime input-conditioned precision selector per layer. | NeurIPS 2025; code available. |
| 3 | [Any-Precision LLM](any_precision_llm_2024.md) | Overlays multiple bitwidth LLMs into one memory image. | ICML 2024 oral; code available. |
| 4 | [QuEPT](quept_2026.md) | One-shot calibrated elastic precision Transformer with real-time bit switching. | AAAI 2026; code available. |
| 5 | [MatGPTQ](matgptq_2026.md) | Single sliceable checkpoint with bit-slicing and mixed-precision kernels. | arXiv 2026; code available. |
| 6 | [NestedFP](nestedfp_2025.md) | FP16/FP8 nested storage for dynamic SLO-aware precision switching. | NeurIPS 2025; code available. |
| 7 | [ARKV](arkv_2026.md) | Token states are Original, Quantized, or Evicted: high-bit, low-bit, or 0-bit. | arXiv 2026; code available. |
| 8 | [Don't Waste Bits](dont_waste_bits_2026.md) | Learned per-token KV bitwidth from importance features. | arXiv 2026; no code found. |
| 9 | [QAQ KV Cache](qaq_kv_cache_2024.md) | Quality-adaptive KV-cache quantization with outlier and attention-aware protection. | 2024 preprint / ICCV workshop; code available. |
| 10 | [SliM-LLM](slim_llm_2024.md) | Salience-driven group-wise LLM bit allocation. | ICML 2025; code available. |

## Best Papers To Read First

### If the proposal is about prompt/token-conditioned precision

Read:

1. MoBiQuant
2. DP-LLM
3. Don't Waste Bits
4. ARKV
5. QAQ KV Cache

These directly support the claim that precision should depend on the current
token, layer state, cache importance, or runtime input.

### If the proposal is about storing different bitwidths

Read:

1. MoBiQuant
2. Any-Precision LLM
3. MatGPTQ
4. NestedFP
5. Matryoshka Quantization

These directly address the storage problem: a deployable system should not keep
separate checkpoints for every bitwidth.

### If the proposal is about soft pruning

Read:

1. Bayesian Bits
2. ARKV
3. MoBiQuant
4. DP-LLM
5. DiffQ

These give the cleanest bridge:

```text
0 bits      = prune / evict
low bits    = coarse keep
high bits   = precise keep
extra slices = refinement
```

## Synthesis

The emerging pattern is:

```text
importance signal -> precision level -> nested/sliceable storage -> hardware-aware execution
```

MoBiQuant is one of the strongest conceptual fits because its router turns token
sensitivity into the number of residual bit slices used. DP-LLM generalizes the
same intuition to layer-wise runtime precision. ARKV and Don't Waste Bits apply
the idea to KV cache. Any-Precision LLM, MatGPTQ, and NestedFP address the
storage/layout side.

## Research Gap

No single reviewed, open-source paper fully combines all of the following:

- prompt/token-conditioned precision,
- weight-side nested or residual bit storage,
- explicit soft-pruning interpretation with 0-bit/low-bit/high-bit states,
- hardware-backed kernels,
- strong peer-reviewed acceptance.

That gap is exactly where this project can be positioned.

