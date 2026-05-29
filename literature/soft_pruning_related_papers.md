# Soft Pruning Related Paper Collection

This note collects papers that are highly related to the idea of "soft pruning":
instead of making an immediate hard 0/1 keep-or-remove decision, the model keeps a
learnable, differentiable, recoverable, or multi-level importance variable during
optimization. For the mixed-precision quantization topic, the closest analogue is
not ordinary pruning, but learnable bit allocation: 0-bit corresponds to pruning,
while 2/4/8/etc. bits correspond to retaining different amounts of information.

## How To Use This For The Current Topic

For the team's selected topic, the clean formulation should avoid claiming that
the work is pruning. The useful bridge is:

- Hard pruning: each block/channel/weight is assigned a binary state, keep or drop.
- Soft pruning: each component has a learnable importance/gate/mask before the
  final hard decision.
- Mixed precision analogue: each layer/group/weight is assigned a bitwidth, so
  information is not simply kept or removed but represented with different
  precision levels.

The most relevant proposal wording is therefore:

> We reformulate binary keep/drop compression as precision allocation. Instead of
> assigning each component a hard 0/1 pruning decision, we learn or search for the
> amount of information that should be preserved through its bitwidth. Components
> that are sensitive to quantization receive higher precision, while redundant or
> tolerant components receive fewer bits. The key challenge is to avoid storing
> multiple quantized versions of the same weight and to ensure the final policy is
> compatible with hardware-supported bitwidths.

## Closest Papers: Quantization As Soft / Multi-Level Pruning

These are the papers most directly related to the user's example where "important
weights keep more bits and less important weights keep fewer bits."

| Priority | Paper | Why It Is Related |
|---|---|---|
| Very high | Bayesian Bits: Unifying Quantization and Pruning | Closest conceptual match. It treats bitwidth as controlled by learnable stochastic gates and includes a 0-bit option, so pruning and mixed-precision quantization become one framework. |
| Very high | DiffQ: Differentiable Model Compression via Pseudo Quantization Noise | Learns the number of bits per individual weight or groups of weights through a differentiable model-size objective. Very close to "softly choose how many bits to keep." |
| Very high | FracBits: Mixed Precision Quantization via Fractional Bit-Widths | Uses fractional bitwidths during optimization, then resolves them into deployable mixed precision. Strong soft-to-hard bitwidth search reference. |
| High | AdaBits: Neural Network Quantization with Adaptive Bit-Widths | Trains one model that can execute at different bitwidths. Useful for the storage/redundancy issue because it targets adaptive precision without training separate models for each bitwidth. |
| High | Matryoshka Quantization | Trains a single nested-precision model so lower bitwidths are contained in higher bitwidth representations. Very relevant to "do not store multiple bitwidth copies." |
| High | Any-Precision Deep Neural Networks | A single runtime model can be set to different bitwidths by truncating least significant bits. Useful for adaptive deployment framing. |
| High | Mixed Precision Quantization via DNAS | Differentiable search over layer-wise bitwidths; useful as the search-space formulation. |
| High | HAQ: Hardware-Aware Automated Quantization with Mixed Precision | Hardware-aware per-layer bitwidth search. Useful for the final deployment constraint. |
| Medium | AutoQ: Automated Kernel-Wise Neural Network Quantization | Searches bitwidths at kernel granularity, showing that importance can vary even within a layer. |
| Medium | Efficient Bitwidth Search for Practical Mixed Precision Neural Network | Important for the storage/memory difficulty: it reuses meta weights across candidate bitwidths instead of keeping many separate copies. |
| Medium | Differentiable Soft Quantization | Not bit allocation, but important for the "soft during training, hard at deployment" quantization mechanism. |

### Bayesian Bits: Unifying Quantization and Pruning

- Authors: Mart van Baalen, Christos Louizos, Markus Nagel, Rana Ali Amjad, Ying Wang, Tijmen Blankevoort, Max Welling
- Venue/year: NeurIPS 2020
- Source: https://papers.nips.cc/paper/2020/hash/3f13cf4ddf6fc50c0d39a1d5aeb57dd8-Abstract.html
- Core idea: Decompose quantization into sequential residual bit additions.
  Learn stochastic gates that decide whether additional bit residuals should be
  included. A 0-bit option unifies pruning with quantization.
- Why it matters here: This is almost exactly the "soft pruning but with bits"
  story. Instead of keeping/dropping weights, the model learns how many bits of
  information each tensor should retain.
- Useful slide angle: "0 bit = prune; 2/4/8 bits = keep increasingly more
  information."

### DiffQ: Differentiable Model Compression via Pseudo Quantization Noise

- Authors: Alexandre Defossez, Yossi Adi, Gabriel Synnaeve
- Year: 2021
- Source: https://arxiv.org/abs/2104.09987
- Core idea: Add pseudo quantization noise during training to approximate the
  effect of quantization while keeping the objective differentiable with respect
  to both weights and bit allocation.
- Why it matters here: The method directly optimizes the number of bits per
  weight or per group. This is a strong precedent for soft bit allocation without
  brute-force enumerating all precision choices.
- Useful slide angle: "Replace discrete bitwidth search with differentiable
  model-size regularization."

### FracBits: Mixed Precision Quantization via Fractional Bit-Widths

- Authors: Linjie Yang, Qing Jin
- Venue/year: AAAI 2021
- Source: https://ojs.aaai.org/index.php/AAAI/article/view/17269
- Core idea: During QAT, each layer/kernel can have a fractional bitwidth between
  two neighboring bitwidths. Differentiable regularization pushes the final model
  toward a deployable mixed-precision policy under resource constraints.
- Why it matters here: Fractional bitwidth is a very clean "soft" version of
  discrete precision assignment.
- Useful slide angle: "Soft bitwidth during optimization, hard hardware-friendly
  bitwidth after search."

### AdaBits: Neural Network Quantization With Adaptive Bit-Widths

- Authors: Qing Jin, Linjie Yang, Zhenyu Liao
- Venue/year: CVPR 2020
- Source: https://arxiv.org/abs/1912.09666
- Core idea: Train a quantized neural network that can adaptively execute at
  multiple bitwidths for weights and activations.
- Why it matters here: It directly addresses the problem that separate models for
  separate bitwidths create storage and training redundancy.
- Useful slide angle: "One trained model supports multiple deployment budgets."

### Matryoshka Quantization

- Authors: Pranav Nair, Puranjay Datta, Jeff Dean, Prateek Jain, Aditya Kusupati
- Year: 2025
- Source: https://arxiv.org/abs/2502.06786
- Core idea: Use the nested structure of integer representations so one quantized
  model can be served at multiple precision levels. Lower precision is nested
  inside higher precision.
- Why it matters here: This is the strongest recent answer to "we cannot store
  multiple bitwidth versions of the same weight."
- Useful slide angle: "Store once, serve at multiple precisions."

### Any-Precision Deep Neural Networks

- Authors: Huanrui Yang et al.
- Venue/year: AAAI 2021
- Source: https://ojs.aaai.org/index.php/AAAI/article/view/17286
- Core idea: A runtime model can flexibly switch bitwidths by truncating least
  significant bits.
- Why it matters here: It supports the same multi-precision single-model story as
  AdaBits and Matryoshka Quantization.
- Useful slide angle: "Precision becomes a runtime knob rather than a separate
  checkpoint."

### Mixed Precision Quantization of ConvNets via DNAS

- Authors: Bichen Wu, Yanghan Wang, Peizhao Zhang, Yuandong Tian, Peter Vajda, Kurt Keutzer
- Year: 2018
- Source: https://arxiv.org/abs/1812.00090
- Core idea: Formulate layer-wise mixed precision as differentiable architecture
  search, replacing exponential discrete search with gradient-based optimization.
- Why it matters here: It is one of the clearest formulations of bitwidth
  assignment as a searchable design space.
- Useful slide angle: "Precision search is a NAS problem over bitwidth choices."

### HAQ: Hardware-Aware Automated Quantization With Mixed Precision

- Authors: Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, Song Han
- Venue/year: CVPR 2019
- Source: https://arxiv.org/abs/1811.08886
- Core idea: Use reinforcement learning and direct hardware feedback to select
  layer-wise bitwidth under latency, energy, and model-size constraints.
- Why it matters here: It is the strongest reference for the hardware-aware part.
  Bit allocation should be chosen under real hardware cost, not just abstract
  compression ratio.
- Useful slide angle: "Different hardware targets prefer different precision
  policies."

### AutoQ: Automated Kernel-Wise Neural Network Quantization

- Authors: Qian Lou, Feng Guo, Lantao Liu, Minje Kim, Lei Jiang
- Year: 2019
- Source: https://arxiv.org/abs/1902.05690
- Core idea: Search quantization bit number per convolution kernel and per
  activation layer using hierarchical reinforcement learning.
- Why it matters here: It shows that precision importance can vary at finer
  granularity than the whole layer.
- Useful slide angle: "Not only layers, even kernels can deserve different bits."

### Efficient Bitwidth Search for Practical Mixed Precision Neural Network

- Authors: Zhen Dong et al.
- Year: 2020
- Source: https://arxiv.org/abs/2003.07577
- Core idea: Efficient bitwidth search that reuses meta weights for different
  candidate precisions to reduce memory and computational overhead.
- Why it matters here: This directly matches the practical difficulty that we
  cannot store all candidate bitwidth versions of every weight.
- Useful slide angle: "Search without duplicating every candidate model."

### Differentiable Soft Quantization

- Authors: Ruihao Gong, Xianglong Liu, Shenghu Jiang, Tianxiang Li, Peng Hu, Jiazhen Lin, Fengwei Yu, Junjie Yan
- Venue/year: ICCV 2019
- Source: https://arxiv.org/abs/1908.05033
- Core idea: Replace hard quantization during training with a differentiable soft
  approximation that gradually evolves toward standard quantization.
- Why it matters here: It does not solve bit allocation by itself, but it is a
  useful mechanism reference for "soft during training, hard at deployment."
- Useful slide angle: "Quantization can be relaxed during training to make
  optimization smoother."

## Direct Soft Pruning / Differentiable Mask Papers

These papers are useful for explaining the analogy, but they should not become
the main project topic if pruning has already been assigned to another teammate.

| Priority | Paper | Why It Is Related |
|---|---|---|
| Very high | Soft Filter Pruning for Accelerating Deep CNNs | Canonical "soft pruning" term: pruned filters are zeroed but can still be updated and recover during training. |
| High | Asymptotic Soft Filter Pruning | Improves SFP by changing pruning rate over training. |
| High | Learning Sparse Neural Networks through L0 Regularization | Learnable stochastic gates provide a differentiable relaxation of binary pruning. |
| High | Network Slimming | Learns channel importance via sparse scaling factors, then removes small channels. |
| High | Movement Pruning | Transformer/NLP-focused adaptive pruning during fine-tuning using movement/first-order information. |
| High | CoFi: Structured Pruning Learns Compact and Accurate Models | Transformer structured pruning with masks at multiple granularities and distillation. |
| Medium | Differentiable Subset Pruning of Transformer Heads | Uses differentiable subset selection for attention head pruning. |
| Medium | Operation-Aware Soft Channel Pruning using Differentiable Masks | Learns soft channel masks during optimization. |
| Medium | Differentiable Network Pruning via Polarization of Probabilistic Channelwise Soft Masks | Uses probabilistic soft masks and polarization regularization. |
| Medium | PDP: Parameter-free Differentiable Pruning | Recent differentiable pruning method emphasizing soft mask recovery. |
| Medium | MaskLLM: Learnable Semi-Structured Sparsity for LLMs | LLM-specific learnable N:M sparsity with Gumbel-Softmax sampling. |

### Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks

- Authors: Yang He, Guoliang Kang, Xuanyi Dong, Yanwei Fu, Yi Yang
- Venue/year: IJCAI 2018
- Source: https://www.ijcai.org/proceedings/2018/309
- Core idea: Instead of permanently removing pruned filters during training, set
  them to zero but still allow them to receive gradient updates. Final pruning is
  applied after training.
- Why it matters here: This is the cleanest reference for the exact phrase "soft
  pruning."
- Relation to quantization topic: The analogue is to avoid immediate irreversible
  binary decisions; keep a soft/learnable state while optimizing the final compact
  model.

### Asymptotic Soft Filter Pruning

- Authors: Yang He, Xuanyi Dong, Guoliang Kang, Yanwei Fu, Chenggang Yan, Yi Yang
- Year: 2018/2019
- Source: https://arxiv.org/abs/1808.07471
- Core idea: Extends SFP with an asymptotic pruning schedule so the pruning ratio
  increases over training.
- Why it matters here: Useful if the proposal needs a staged soft-to-hard
  schedule.

### Learning Sparse Neural Networks through L0 Regularization

- Authors: Christos Louizos, Max Welling, Diederik P. Kingma
- Venue/year: ICLR 2018
- Source: https://openreview.net/forum?id=H1Y8hhg0b
- Core idea: Uses a differentiable stochastic gate relaxation to optimize L0-like
  sparsity with gradient descent.
- Why it matters here: It is foundational for learnable gates and differentiable
  binary pruning decisions.
- Relation to quantization topic: Bayesian Bits builds directly on this style of
  stochastic gating, but applies it to bitwidth decisions.

### Learning Efficient Convolutional Networks through Network Slimming

- Authors: Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang
- Venue/year: ICCV 2017
- Source: https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html
- Core idea: Learn sparse scaling factors on batch normalization channels; small
  scaling factors indicate channels that can be pruned.
- Why it matters here: It is a classic importance-signal-based pruning paper.
  Unlike hard magnitude pruning, importance is learned during training.

### Movement Pruning: Adaptive Sparsity by Fine-Tuning

- Authors: Victor Sanh, Thomas Wolf, Alexander M. Rush
- Venue/year: NeurIPS 2020
- Source: https://proceedings.neurips.cc/paper/2020/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html
- Core idea: During pretrained model fine-tuning, decide pruning by whether
  weights move toward or away from zero, rather than only by magnitude.
- Why it matters here: This is a strong Transformer-related pruning reference.
  It is less directly tied to bitwidth, but relevant to sensitivity/importance
  estimation during adaptation.

### Structured Pruning Learns Compact and Accurate Models

- Authors: Mengzhou Xia, Zexuan Zhong, Danqi Chen
- Venue/year: ACL 2022
- Source: https://arxiv.org/abs/2204.00408
- Core idea: CoFi jointly prunes Transformer components at coarse and fine
  granularity, including layers, heads, and hidden dimensions, using masks and
  layerwise distillation.
- Why it matters here: It is a useful language-model compression reference if the
  discussion needs Transformer-specific soft/structured pruning background.

### Differentiable Subset Pruning of Transformer Heads

- Authors: Elena Voita et al. / related TACL work
- Venue/year: TACL 2021
- Source: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00436/108868/Differentiable-Subset-Pruning-of-Transformer-Heads
- Core idea: Uses differentiable subset selection / L0-style gates for attention
  head pruning.
- Why it matters here: Shows how a discrete keep/drop decision over Transformer
  heads can be relaxed for optimization.

### Operation-Aware Soft Channel Pruning using Differentiable Masks

- Authors: Zhuangwei Zhuang et al.
- Year: 2020
- Source: https://arxiv.org/abs/2007.03938
- Core idea: Learns differentiable masks for individual channels and keeps soft
  decisions throughout optimization.
- Why it matters here: Good reference for soft masks at channel granularity.

### Differentiable Network Pruning via Polarization of Probabilistic Channelwise Soft Masks

- Authors: Shoukai Yu et al.
- Year: 2021/2022
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC9098282/
- Core idea: Learns probabilistic channelwise soft masks and uses polarization
  regularization to push masks toward pruned or kept states.
- Why it matters here: Useful for explaining soft-to-hard mask polarization.

### PDP: Parameter-free Differentiable Pruning

- Authors: NeurIPS 2023 paper
- Source: https://proceedings.neurips.cc/paper_files/paper/2023/file/8f9f4eb32b9081a90f2a0b2627eb2a24-Paper-Conference.pdf
- Core idea: Differentiable pruning with soft masks that can recover from being
  pruned during training.
- Why it matters here: Recent differentiable pruning reference; good if the
  survey wants a modern "soft mask can recover" example.

### MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models

- Year: 2024
- Source: https://arxiv.org/abs/2409.17481
- Core idea: Uses learnable masks and Gumbel-Softmax sampling to establish N:M
  semi-structured sparsity in LLMs.
- Why it matters here: LLM-scale learnable sparsity reference. Keep as background
  only because pruning has been assigned elsewhere.

## Recommended Reading Order

If the goal is to support a mixed-precision proposal inspired by soft pruning:

1. Bayesian Bits
2. DiffQ
3. FracBits
4. AdaBits
5. Matryoshka Quantization
6. HAQ
7. DNAS mixed precision quantization
8. Efficient Bitwidth Search
9. Soft Filter Pruning
10. L0 Regularization

## 2026-05-30 Focused Refresh: Prompt/Hardware-Aware Adaptive Bits

The newer LLM literature makes the current project easier to position. The
proposal is not just "mixed precision"; it is **adaptive information storage**:
the model should preserve more bits for components or prompts that need them,
but the representation must still be packable and executable.

New paper notes added in this refresh:

- [ScaleBITS](scalebits_2026.md): hardware-aligned fine-grained LLM bitwidth
  search under a memory budget.
- [SliM-LLM](slim_llm_2024.md): salience-driven group-wise mixed precision for
  LLMs.
- [Matryoshka Quantization](matryoshka_quantization_2025.md): nested bit-plane
  storage so a single quantized model can serve multiple precisions.
- [AdaQuantLM](adaquantlm_2024.md): additive codewords for adaptive bitwidths
  without keeping full-precision weights.
- [QAQ](qaq_query_adaptive_quantization_2025.md): query-conditioned precision
  selection over bit planes with CPU/GPU loading trade-offs.
- [Prompt-Adaptive Quantization](prompt_adaptive_quantization_2026.md):
  per-prompt routing to pre-quantized model variants.
- [Channel-Wise Mixed-Precision Quantization for LLMs](channel_wise_mixed_precision_quantization_llm_2024.md):
  activation-distribution-driven channel-wise allocation.
- [FineQ](fineq_2025.md) and [FGMP](fgmp_2025.md): systems papers on alignment,
  packing, outlier protection, and hardware support.
- [RAMP](ramp_2026.md): RL-based LLM bit allocation from activation, weight, and
  structural features.

The most useful synthesis is in
[Focused Index: Soft Pruning as Adaptive Mixed-Precision Information Allocation](soft_pruning_mixed_precision_index_2026-05-30.md).

## Slide Formulation To Reuse

### Pain Point

Existing compression methods often make hard structural decisions: a component is
kept or removed, or the whole model is quantized with one uniform bitwidth. This
is too coarse. Different layers, channels, heads, or weight groups contribute
unequally to model quality, so they should preserve different amounts of
information.

### Core Question

Can we replace binary keep/drop compression with information-preserving precision
allocation, where each component receives the minimum bitwidth needed to preserve
quality under a memory/latency/hardware constraint?

### Main Difficulty

The search space is combinatorial, and a naive implementation that stores every
candidate bitwidth version of every weight would eliminate the memory benefit.
The final policy must also match hardware-supported bitwidths and kernels.

### Candidate Method

Use a soft or differentiable importance variable during search:

```text
full-precision model
  -> calibration / QAT / distillation signals
  -> soft importance or bitwidth score per layer/group
  -> resource-constrained precision assignment
  -> deployable mixed-precision model
```

### Best Conceptual Anchor

Bayesian Bits provides the cleanest bridge:

```text
0 bit      = pruned
low bits   = less information retained
high bits  = more information retained
```

This lets us explain "soft pruning" without actually making pruning the project
topic.
