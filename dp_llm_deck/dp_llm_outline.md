# Slide 1: Problem Setting and Research Gap
**Visual: cards-3**

**Goal: adapt one on-device LLM to changing latency, memory, and quality constraints.**

[Card 1: Past research: static quantization]
- Weight-only PTQ such as OPTQ/GPTQ, AWQ, and SqueezeLLM reduces memory traffic.
- The assigned precision is fixed after calibration.
- It cannot react to per-query runtime constraints.

[Card 2: Past research: multi-scale / mixed precision]
- Any-Precision LLM and Matryoshka Quantization store multiple bit-width views compactly.
- Static layer-wise mixed precision can target non-integer effective bit-widths.
- But configurations are still selected offline or fixed during decoding.

[Card 3: What is missing]
- How should a model match a target precision or latency at runtime?
- Layer sensitivity changes token by token during decoding.
- Static precision leaves useful latency-accuracy trade-offs unused.

Takeaway: DP-LLM treats precision as a runtime, layer-wise decision.

---

# Slide 2: Core Observation: Sensitivity Changes by Token
**Visual: comparison-3**

**Layer sensitivity is not static; the high-precision layers change across decoding steps.**

[Column 1: Uniform]
- Same bit-width for every layer.
- Coarse adaptation set.

[Column 2: Static layer-wise]
- Different layers get different bit-widths.
- Assignment is fixed across all tokens.

[Column 3: Dynamic layer-wise]
- Each layer can switch between low and high precision per token.
- Better matches observed sensitivity changes.

Takeaway: Dynamic assignment can improve perplexity versus static mixed precision in the paper's oracle analysis.

---

# Slide 3: Method: DP-LLM Precision Selector
**Visual: process-4-phase**

**DP-LLM learns per-layer candidate precisions and thresholds offline, then selects bit-widths online.**

[Column 1: Fit memory budget]
- Use static sensitivity to choose each layer's maximum precision.
- Respect the memory budget of the multi-scale quantized model.

[Column 2: Assign average precision]
- Fine-tune one scalar p_i per layer on calibration data.
- Add regularization so the model average matches the target bit-width.

[Column 3: Translate to thresholds]
- Candidate set becomes floor(p_i) and ceil(p_i).
- Threshold T_i is chosen from the calibration relative-error distribution.

[Column 4: Runtime selection]
- Estimate ||(W_h - W_l)x|| cheaply.
- If error exceeds T_i, use high precision; otherwise use low precision.

---

# Slide 4: Main Results and Takeaways
**Visual: cards-3**

**DP-LLM improves the performance-latency trade-off while keeping selector overhead small.**

[Card 1: Quality]
- Evaluated on Llama-3-8B and Phi-3-Medium.
- Lower perplexity than LLM-MQ and HAWQ-V2 on WikiText2 and C4 across most target precisions.
- Often strongest on GSM8K, MBPP, BBH, and MATH.

[Card 2: Runtime overhead]
- Selector overhead geomean: 1.45% for Llama-3-8B and 0.81% for Phi-3-Medium.
- Hybrid plus async estimation reduces RTX 4060Ti overhead to 0.74%, 0.66%, and 0.45% at 3.5/4.0/4.5 effective bits.

[Card 3: Conclusion]
- Relative error is an effective proxy for precision choice.
- Dynamic layer-wise assignment captures token-step sensitivity missed by static baselines.
- Query-level effective-bitwidth drift remains small: 99th percentile is about 2.25-3.32%.

Takeaway: DP-LLM converts target precision into low-overhead per-layer runtime adaptation.
