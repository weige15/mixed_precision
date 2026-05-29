# Slide 1: MoBiQuant: problem setting and research gap

**Visual: comparison-3**

[Column 1: Static PTQ]
- GPTQ / OmniQuant / SmoothQuant / AWQ optimize one chosen bit-width.
- Strong at a fixed precision, but brittle when runtime precision changes.

[Column 2: Any-precision PTQ]
- AnyPrecisionLLM / AnyBCQ / MatQuant support multiple bit-widths.
- They still need extra scaling, repacking, table lookups, or inefficient kernels.

[Column 3: Gap]
- Edge/server workloads face changing latency and memory budgets.
- A single model should switch token precision online without separate models.

---

# Slide 2: Core observation: outlier migration breaks cross-bit generalization

**Visual: data-contrast**

**Observation:** Tokens causing high quantization error are not stable across bit-widths.

- 3-bit calibration used for 4-bit inference raises WikiText2 PPL by 2.65 on LLaMA3-8B.
- Keeping the top 10% 3-bit outlier tokens at 3-bit partially recovers quality.
- MoBiQuant routes tokens by sensitivity and reaches 7.31 PPL, close to 4-bit calibration.

---

# Slide 3: Method: many-in-one bit slices plus token-aware routing

**Visual: process-3-phase**

[Column 1: MoBiSlice]
- Recursively quantize residual weights into fixed 2-bit slices.
- Reconstruct 2/4/6/8-bit weights by summing slices.

[Column 2: MoBiRoute]
- A lightweight 2-layer MLP scores each token and bit slice.
- Learned binary gates choose how many slices each token receives.

[Column 3: Kernel path]
- Bit-major packing fetches only active slices.
- Shared scaling and fused routing reduce bandwidth and launch overhead.

---

# Slide 4: Main results and conclusion

**Visual: cards-4**

[Card 1: Accuracy]
- Matches or surpasses static scalar PTQ on LLaMA2/LLaMA3 families.
- Large gains at 2-3 bit when compared with any-precision baselines.

[Card 2: Throughput]
- Average decoding speedup: 33.8% over AnyPrecisionLLM.
- Average decoding speedup: 22.8% over AnyBCQ.

[Card 3: Memory]
- One nested model replaces multiple per-bit deployments.
- Reported memory footprint reduction is up to 3.5x.

[Card 4: Takeaway]
- Precision becomes a token-level runtime resource.
- The main mechanism is mitigating outlier migration, not only storing more bits.

---
