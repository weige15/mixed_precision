# Slide 1: MatQuant: problem setting and research gap
**Visual: comparison-3**

[Column 1: Deployment bottleneck]
- LLM decoding is often memory-bandwidth bound because weights must be moved repeatedly.
- Quantization reduces communication and inference cost.
- Low precision such as int2 can severely damage model quality.

[Column 2: Past research direction]
- MinMax, GPTQ, AWQ, SmoothQuant, OmniQuant, and QAT optimize one target precision.
- Multi-scale / nested models were explored, but mostly outside modern LLM weight quantization.
- Serving different bit-widths usually means maintaining separate checkpoints.

[Column 3: What was missing]
- One trained model that can be served at int8, int4, int2, and intermediate precisions.
- Low-bit quality that does not collapse when sliced from a high-bit model.
- A dense accuracy-cost trade-off without retraining each point.

---

# Slide 2: Core idea: use the nested structure of integer bits
**Visual: process-3-phase**

[Column 1: Train a parent]
- Quantize into a largest integer precision such as int8.
- Treat the most significant bits as shared information.

[Column 2: Slice lower precision]
- Extract int4 or int2 by taking only the most significant bits.
- The smaller model is nested inside the larger model.

[Column 3: Joint objective]
- Optimize losses for several bit-widths at once.
- MatQuant is added on top of learning-based methods such as OmniQuant and QAT.

---

# Slide 3: Observations: why MatQuant helps low-bit models
**Visual: cards-4**

[Card 1: Shared MSBs matter]
- Directly slicing a normal int8 model performs poorly.
- MatQuant trains the shared MSBs to carry useful low-bit information.

[Card 2: Distribution shift]
- MatQuant shifts quantized weight usage toward higher values.
- This helps int2 use more of its very small bucket set.

[Card 3: Loss weighting]
- Higher weight on the int2 loss is important.
- Increasing int4/int8 emphasis can hurt int2.

[Card 4: Co-distillation]
- Lower-precision slices can learn from higher-precision slices.
- The nested structure creates a natural teacher-student relation inside one model.

---

# Slide 4: Results and takeaway
**Visual: cards-4**

[Card 1: Accuracy]
- int8 and int4 stay within about 0.5% of separately trained baselines in OmniQuant experiments.
- int2 improves by 1.04%, 3.11%, and 3.01% on Gemma-2 2B, Gemma-2 9B, and Mistral 7B.

[Card 2: Generality]
- With QAT, int2 gains reach 4.46%, 6.27%, and 7.02% on the same model families.
- The method applies to any learning-based quantization path.

[Card 3: Elastic serving]
- int3 and int6 can be obtained by slicing, even when not directly optimized.
- Layer-wise Mix'n'Match gives dense accuracy-vs-cost trade-offs.

[Card 4: Takeaway]
- MatQuant turns precision into a deploy-time choice.
- The main contribution is co-optimizing nested bit slices, not merely storing an int8 checkpoint.

