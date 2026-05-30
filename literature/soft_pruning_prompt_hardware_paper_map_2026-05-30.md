# Paper Map: Soft Pruning as Prompt- and Hardware-Aware Mixed Precision

## User Idea

Treat "soft pruning" as variable information allocation rather than binary
removal. A weight, channel, block, layer, or cache entry can receive:

- 0 bits: prune, evict, or ignore.
- Low bits: keep coarse information.
- High bits: keep sensitive information.
- Residual bit slices or bit planes: add refinement only when needed.

The proposed policy is:

```text
bitwidth = f(component importance, prompt difficulty, hardware cost)
```

The central systems difficulty is storage and execution: arbitrary per-weight
bitwidths create packing, metadata, alignment, and kernel problems. The strongest
papers avoid fully irregular storage by using groups, blocks, bit planes,
residual slices, sparse exceptions, or hardware-aligned layouts.

## Highest-Relevance Papers

| Role | Paper | Why it matters |
|---|---|---|
| Soft-pruning bridge | Bayesian Bits: Unifying Quantization and Pruning (NeurIPS 2020) | Explicitly unifies 0-bit pruning with higher-bit quantization through residual bit additions and stochastic gates. This is the cleanest conceptual reference for "soft pruning = learned bit allocation." |
| Prompt/token-conditioned precision | QAQ: Query-adaptive Mixed-precision Quantization for LLMs (NeurIPS 2025 ML for Systems workshop) | Decomposes weights into bit planes and uses a query-conditioned router, with on-demand CPU/GPU loading. Direct match to "input prompt decides precision." |
| Token-adaptive residual storage | MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM (arXiv 2026) | Uses recursive residual quantization and a token-aware router to reconstruct higher precision at runtime. Direct match to "store base plus optional refinement." |
| Storage-efficient multi-precision | Matryoshka Quantization (ICML 2025 / arXiv 2025) | Uses nested integer precision so one model can be served at multiple bitwidths instead of storing many checkpoints. Strong answer to the storage difficulty. |
| Salience-based LLM bit allocation | SliM-LLM: Salience-Driven Mixed-Precision Quantization for LLMs (ICML 2025) | Assigns more bits to salient weight groups, making importance-to-bitwidth concrete for LLM weights. |
| Hardware-aware assignment | HAQ: Hardware-Aware Automated Quantization with Mixed Precision (CVPR 2019) | Classic hardware feedback loop for layer bitwidths under latency, energy, and model-size constraints. |
| Hardware-aligned LLM allocation | ScaleBITS: Scalable Bitwidth Search for Hardware-Aligned Mixed-Precision LLMs (arXiv 2026) | Formulates global bitwidth allocation under memory budget with hardware-aligned block partitioning and channel reordering. |
| Hardware/software co-design | FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference (arXiv 2025) | Selects high-precision weight and activation blocks using perturbation weighted by Fisher information, then proposes hardware support for block mixed precision. |
| Sparse high-precision exceptions | SpQR, OWQ, SqueezeLLM (2023) | These start from a low-bit representation and store sensitive/outlier weights in higher precision. They are practical precedents for "low-bit base plus high-bit rescue." |
| Learned hardware/input feature policy | RAMP: Reinforcement Adaptive Mixed Precision Quantization (arXiv 2026) | Learns per-layer bitwidths from activation statistics, weight properties, and structure under a global bit budget; useful but preprint-only. |

## Papers Grouped By The User's Three Importance Sources

### Importance From The Input Prompt

- QAQ: query-conditioned router chooses precision from bit planes.
- MoBiQuant: token-aware router chooses residual bit slices.
- DP-LLM: input-conditioned layer precision selector.
- Instance-Aware Dynamic Neural Network Quantization: reviewed precedent that
  individual inputs can choose different bitwidths, though vision-focused.
- ARKV / Don't Waste Bits / QAQ KV Cache: prompt or token state decides KV-cache
  precision or eviction, giving a clean 0-bit/low-bit/high-bit analogy.

### Importance From Hardware

- HAQ: hardware simulator feedback chooses layer precision.
- ScaleBITS: hardware-aligned block layout and constrained bitwidth allocation.
- FineQ and FGMP: co-design mixed precision with aligned data movement and
  datapath support.
- QServe, Ladder/BitBLAS, TorchAO, bitsandbytes, and Transformer Engine:
  practical reminders that a bitwidth policy only saves resources when kernels
  and layouts support it.

### Importance From Both Prompt And Hardware

- QAQ: prompt-conditioned precision plus CPU/GPU loading trade-off.
- MoBiQuant: token-conditioned residual slices for elastic runtime budgets.
- RAMP: conditions on activation/weight/structure features and exports to GGUF
  for device deployment.
- DP-LLM / Any-Precision LLM / NestedFP / MatGPTQ: closest cluster for dynamic
  or elastic precision with a single checkpoint or shared storage.

## Storage Design Lessons

The naive design, "each weight can have any bitwidth," is usually not deployable.
It fragments memory and adds metadata that can erase compression gains. The
better design space is structured:

- Nested bit planes: Matryoshka Quantization, QAQ, Any-Precision LLM.
- Residual bit slices: MoBiQuant, MatGPTQ.
- Additive codewords: AdaQuantLM / AQLM-style references.
- Sparse high-precision exceptions: SpQR, OWQ, SqueezeLLM.
- Block/group/channel bitwidths: SliM-LLM, ScaleBITS, FineQ, FGMP.
- Cache states as precision states: ARKV and related KV-cache work.

## Best Research Direction

A defensible next hypothesis is:

```text
Weights can be decomposed into a low-bit base plus structured refinement
components. A router activates refinement only for sensitive modules, difficult
prompts, or hardware states where the extra precision is worth the memory and
latency cost.
```

This combines the user's disentanglement idea with the storage lesson from the
literature. The model should not store independent 2-bit, 4-bit, 8-bit, and
16-bit copies. It should store nested or residual information once, then expose a
small number of hardware-aligned precision choices.

## Recommended Reading Order

1. Bayesian Bits - conceptual foundation for soft pruning as bit allocation.
2. Matryoshka Quantization - storage-efficient single-model multi-precision.
3. QAQ and MoBiQuant - prompt/token-conditioned precision routing.
4. SliM-LLM and ScaleBITS - salience and hardware-aligned LLM bit allocation.
5. SpQR / OWQ / SqueezeLLM - sparse high-precision rescue formats.
6. HAQ / FGMP / FineQ - hardware-aware optimization and kernel implications.

## Source Links

- Bayesian Bits: https://arxiv.org/abs/2005.07093
- Matryoshka Quantization: https://arxiv.org/abs/2502.06786
- QAQ: https://neurips.cc/virtual/2025/129098
- MoBiQuant: https://arxiv.org/abs/2602.20191
- SliM-LLM: https://arxiv.org/abs/2405.14917
- HAQ: https://arxiv.org/abs/1811.08886
- ScaleBITS: https://arxiv.org/abs/2602.17698
- FGMP: https://arxiv.org/abs/2504.14152
- SpQR: https://arxiv.org/abs/2306.03078
- OWQ: https://arxiv.org/abs/2306.02272
- SqueezeLLM: https://arxiv.org/abs/2306.07629
- RAMP: https://arxiv.org/abs/2603.17891
