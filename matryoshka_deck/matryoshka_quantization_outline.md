# Slide 1: Problem: multi-precision deployment still behaves like multi-model deployment

**Visual: comparison-3**

**Matryoshka Quantization targets the maintenance and quality gap created when each bit-width is optimized separately.**

[Column 1: Deployment pressure]
- LLM decoding is often memory-bandwidth limited, so weight movement from HBM/SRAM becomes a major inference cost.
- Weight quantization reduces communication cost and makes int8, int4, and int2 attractive serving formats.

[Column 2: Prior research directions]
- Learning-free PTQ: MinMax, GPTQ, AWQ, SmoothQuant, QuIP, QuaRoT reduce calibration/training cost.
- Learning-based quantization: QAT, OmniQuant, SpinQuant improve low-bit accuracy by optimizing weights or auxiliary parameters.

[Column 3: What was missing]
- Each target precision is usually treated as an independent optimization problem.
- This leaves teams storing several checkpoints or accepting one fixed accuracy-latency trade-off; int2 remains especially fragile.

Source: Nair et al., "Matryoshka Quantization", ICML 2025 / arXiv:2502.06786v3.

---

# Slide 2: Core idea: train one parent integer model whose MSBs are useful slices

**Visual: process-3-phase**

**MatQuant uses the natural nested structure of integer bit planes: lower precision models are extracted from the most significant bits of the same quantized weights.**

[Column 1: Nest]
- Start from a higher-precision integer representation, e.g. int8.
- Slice the most significant bits to obtain int4 or int2 without storing a separate model.

[Column 2: Jointly optimize]
- During training, optimize losses for multiple target precisions together.
- Default targets are R = {8, 4, 2}; each slice contributes a weighted loss term.

[Column 3: Attach to base quantizers]
- MatQuant is not a replacement for QAT or OmniQuant.
- It wraps learning-based quantizers so one checkpoint can serve several precision budgets.

Key mechanism: shared MSBs are forced to carry information that remains useful after slicing.

---

# Slide 3: Observation: co-training changes how quantization buckets are used

**Visual: cards-3**

**The paper's main empirical observation is that MatQuant is not just truncating int8; training reshapes the quantized weight distribution.**

[Column 1: Right-shifted buckets]
- MatQuant shifts quantized weights toward higher-valued buckets.
- int8 and int4 have enough buckets that small shifts do not strongly hurt quality.

[Column 2: Why int2 benefits]
- int2 has only four buckets, so bucket usage is much more sensitive.
- Joint training uses freedom in the int8 representation to make the int2 slice less under-expressive.

[Column 3: Extra precision for outliers]
- The paper adds an "extra precision" variant around 2.05 effective bits.
- A small extra outlier bucket helps the lowest precision case much more than higher precision cases.

Implication: nested bit structure exists in integers, but the model must be trained to exploit it.

---

# Slide 4: Results and takeaway: one checkpoint covers a denser accuracy-cost curve

**Visual: table**

**MatQuant keeps high-bit quality near independently trained baselines while improving the lowest-bit slice, which is the hard case.**

| Setting | Reported result | Meaning |
|---|---:|---|
| OmniQuant int8 / int4 | Within about 0.5% of independently trained baselines | Little penalty for sharing parameters |
| OmniQuant int2 | +1.04 / +3.11 / +3.01 task average on Gemma-2 2B / 9B / Mistral 7B | Low-cost auxiliary-parameter training still improves low-bit slices |
| QAT int2 | +4.46 / +6.27 / +7.02 task average on the same models | End-to-end learning gives larger gains when data quality is sufficient |
| int6 / int3 slicing | Comparable to explicitly trained baselines | Intermediate bit-widths can be obtained without extra training |
| Mix'n'Match by layer | Dense accuracy-vs-bits trade-off from int2 / int4 / int8 layers | One checkpoint can adapt to different hardware and latency budgets |

Bottom line: MatQuant reframes multi-precision quantization from "train and serve many models" to "train one nested parent model and slice it at deployment time."

