# Calibration-Guided Precision Assignment for LoRA Fine-Tuning

Draft date: 2026-05-20

> Scope note, 2026-05-27: the H10 HAQ-aligned direction has been re-scoped to
> inference-side mixed-precision post-training quantization. The LoRA/QLoRA
> selective-rescue material in this draft should be treated as supporting PEFT
> evidence, not as the main project-plan-aligned H10 contribution.

## Abstract

Low-precision training recipes for large language model adaptation are often selected by broad backend defaults, such as bf16 autocast or QLoRA, rather than by measuring which model operations are actually sensitive to reduced precision. This work studies whether a short pre-training precision check can identify precision-sensitive and precision-tolerant Transformer modules before LoRA fine-tuning. On Qwen2.5 and Llama-3.1 LoRA fine-tuning with Alpaca-style data, we evaluate calibration signals, one-module perturbation probes, frozen mixed-precision policies, and a hardware-aware assignment table. We find that activation-outlier scores predict projection-output perturbation sensitivity, especially for MLP projections, and that rank/perturbation-selected fake-int8 policies preserve bf16 validation quality during actual LoRA updates at both 0.5B and 7B scale. Hardware-backed QLoRA/NF4 gives real memory savings but a quality-throughput trade-off. On Llama-3.1-8B, selective bf16 projection rescue from QLoRA/NF4 improves final eval loss versus blanket QLoRA on every matched RTX 3090 seed while preserving about 25% peak-memory saving versus bf16. For the project-plan-aligned H10 setting, we also instantiate the same table-and-solver principle for inference-side PTQ: on matched Llama-3.1-8B-Instruct vLLM workloads, a GPTQ-Marlin artifact passes a strict 1% prompt-NLL gate and improves latency by about 60-63% with output-throughput gains of about 153-168%.

## 1. Introduction

Mixed precision is now a default part of large language model training and fine-tuning. In practice, however, precision choices are often coarse: users choose bf16, fp16, QLoRA, or an 8-bit backend for an entire run. This hides a more structured problem. Transformer blocks contain operations with different numerical roles: attention projections, MLP projections, normalization, residual paths, logits, losses, optimizer states, and gradients. A single global dtype policy may waste precision on tolerant operations while failing to protect sensitive ones.

The central question in this project is:

> Before LoRA fine-tuning, can a short precision check identify which modules are precision-sensitive, so a frozen mixed-precision policy can match bf16 validation quality while saving resources or expanding the stable fine-tuning envelope?

We study this question in a resource-constrained LoRA setting rather than full pretraining. LoRA makes the experiment grid feasible: multiple seeds, module probes, perturbation tests, and precision policies can be evaluated on available GPUs while keeping the base model fixed. This isolates the precision-assignment question from the much larger cost and confounds of full-model training.

Our findings support a narrower but useful answer. A short calibration and perturbation workflow can identify projection modules that are safe or unsafe for fake-int8 output quantization, and policies selected from those measurements preserve bf16-quality LoRA updates across seeds. The current selective implementation does not yet provide hardware resource savings because it uses Python-level fake-quant hooks. In contrast, generic QLoRA provides a real 7B memory-capacity trade-off, but it is slower than bf16 and does not solve selective precision assignment.

## 2. Problem Formulation

We view precision placement as a constrained assignment problem. Given a model, dataset, hardware target, candidate precision formats, and quality budget, the goal is to assign a precision format to each operation or module:

```text
Input:
  model M
  fine-tuning data D
  hardware target H
  candidate formats F
  quality budget epsilon
  resource objective C

Output:
  precision assignment a_i for each module or operation i

Goal:
  minimize resource cost C(a)
  subject to quality degradation <= epsilon
```

The brute-force search space is combinatorial. If there are `n` decision sites and `p` possible precision formats, a naive search has `p^n` assignments. If the budget is "choose k modules to demote," the search still contains combinatorial `n choose k` possibilities. The research problem is therefore not to enumerate precision recipes, but to predict or estimate module-format risk cheaply enough to guide assignment.

This draft focuses on two tractable versions:

- Model families: Qwen2.5 and Llama-3.1.
- Adaptation method: LoRA / QLoRA-style fine-tuning.
- Primary task metric: held-out validation negative log likelihood.
- Candidate sensitivity intervention: fake-int8 output quantization on selected modules.
- Candidate backend-real intervention: QLoRA/NF4 plus selected bf16 projection rescue.
- Quality gate: final eval loss degradation must remain below 1% relative to matched bf16.
- Stability checks: loss spikes, NaN/Inf events, max gradient norm.
- Resource checks: peak CUDA memory and training tokens/sec.

## 3. Method

The method has four stages.

### 3.1 Calibration Signals

Before training, we run a short calibration pass and collect module-level statistics. The main signals are:

- activation outlier score,
- fake-quantization relative MSE,
- NaN/Inf incidence,
- module role,
- layer position,
- policy trace under conservative thresholds.

The intent is not to trust the thresholded policy directly. Instead, calibration creates a first ranking of likely sensitive and likely tolerant modules.

### 3.2 One-Module Perturbation Probes

Calibration signals are validated by one-module fake-int8 output perturbation probes. For each candidate module, we quantize only that module's output during a forward pass and measure the change in loss. This produces a local perturbation delta:

```text
delta_i = loss(fake_int8_output(module_i)) - loss(bf16)
```

Large positive deltas indicate sensitivity. Near-zero deltas indicate plausible low-precision tolerance. These probes are cheap compared with full fine-tuning and help prevent over-trusting raw quantization-error metrics.

### 3.3 Frozen Policy Training

After calibration and perturbation, we freeze a selected policy before training. This is important: the final comparison must test whether the pre-training precision check predicts training-time behavior, rather than adaptively tuning the policy after seeing validation results.

We compare each frozen policy against a matched bf16 LoRA baseline with the same seed, data size, sequence length, learning rate, effective batch size, and evaluation budget.

### 3.4 Backend-Aware Assignment

Fake-int8 hooks are useful for sensitivity testing, but they are not a resource-saving implementation. To connect sensitivity ranking to real hardware behavior, we adapt the HAQ principle to PEFT. The original HAQ insight is not tied to reinforcement learning itself; the transferable idea is that precision assignment should be chosen under hardware feedback rather than global dtype defaults.

For PEFT, the action table contains one row per `(module_or_group, action, backend, hardware)` candidate:

```text
module_or_group
candidate_action
backend
hardware_label
backend_feasible
predicted_quality_risk
predicted_instability_risk
memory_delta_vs_bf16
throughput_delta_vs_bf16
```

We then solve a small constrained assignment:

```text
choose one feasible action per group
minimize predicted_quality_risk - alpha * quality_recovery
subject to predicted_quality_risk <= epsilon
           predicted_instability_risk <= tau
           memory_saving >= target
```

The first backend-real action space starts from QLoRA/NF4 and optionally rescues selected packed projection modules by reloading their weights in bf16 before LoRA wrapping. This is a selective-rescue policy: keep the low-bit backend for most modules, but protect modules whose target perturbation probes indicate high low-bit risk.

## 4. Experiments

### 4.1 Static fp32 Norm Baseline

We first tested whether hand-picked fp32 normalization islands improve bf16 LoRA. On Qwen2.5-0.5B, seed 42 showed only a 0.20% validation-loss improvement over bf16, below the locked 1% support threshold, with zero loss spikes and zero NaN/Inf events in both policies.

An internal dtype probe explains why this static intervention is weak. The installed Qwen2RMSNorm implementation casts hidden states to fp32 before `pow`, `mean`, and `rsqrt`, then casts back to the input dtype. Thus bf16 boundary tensors do not imply bf16 normalization reduction arithmetic. Static fp32 norm wrapping is therefore not a compelling core contribution in this setup.

### 4.2 0.5B Calibration and Perturbation

At 0.5B, Stage 1 calibration across seeds 42, 43, and 44 observed 218 candidate modules. Policy decisions were stable across seeds for 217 of 218 common modules. The strongest high-risk modules were early or mid MLP down projections with large activation outliers.

Stage 2 perturbation probes supported the predictive value of activation outliers for MLP projections. High-risk MLP down projections had large positive int8 perturbation deltas across seeds:

| Module | Mean int8 loss delta |
|---|---:|
| `layers.2.mlp.down_proj` | +0.3699 |
| `layers.21.mlp.down_proj` | +0.2665 |
| `layers.3.mlp.down_proj` | +0.2209 |

Low-risk late-layer MLP gate/up candidates stayed near zero:

| Module | Mean abs loss delta | Max abs mean delta |
|---|---:|---:|
| `layers.23.mlp.up_proj` | 0.0031 | 0.0070 |
| `layers.23.mlp.gate_proj` | 0.0035 | 0.0048 |
| `layers.22.mlp.up_proj` | 0.0024 | 0.0042 |
| `layers.22.mlp.gate_proj` | 0.0016 | 0.0034 |

Across MLP projection perturbations, max outlier score correlated with absolute loss delta at approximately 0.91. Output int8 relative MSE was weaker and less stable.

### 4.3 0.5B Frozen Policies

The first narrow H6 policy fake-int8 quantized four selected late-layer MLP gate/up modules. Across seeds 42, 43, and 44, it preserved bf16 validation quality:

| Seed | BF16 eval | H6 eval | Relative delta |
|---:|---:|---:|---:|
| 42 | 1.62949 | 1.63112 | +0.100% |
| 43 | 1.63444 | 1.63621 | +0.108% |
| 44 | 1.63247 | 1.63493 | +0.151% |

All runs had zero loss spikes and zero NaN/Inf events.

We then expanded the selected set with a SNIP-style ranking over eligible MLP gate/up modules. The widest tested 0.5B policy, `k=24`, also held across three seeds:

| Seed | BF16 eval | k=24 eval | Relative delta |
|---:|---:|---:|---:|
| 42 | 1.62978 | 1.63177 | +0.122% |
| 43 | 1.61701 | 1.61975 | +0.169% |
| 44 | 1.62089 | 1.62248 | +0.098% |

This supports the claim that calibration-guided scoring can widen the safe fake-int8 candidate set while preserving quality and stability.

### 4.4 Hardware-Backed Low-Bit Baselines

We tested whether existing bitsandbytes low-bit paths provide real resource gains. At 0.5B on RTX 3090, they did not:

| Policy | Eval delta vs bf16 | Memory delta | Throughput delta |
|---|---:|---:|---:|
| fake-int8 k=24 | +0.268% | +0.20% | -2.16% |
| bitsandbytes 8-bit LoRA | +0.102% | +16.80% | -40.16% |
| QLoRA 4-bit NF4 | +2.832% | +15.79% | -37.58% |

At 7B, QLoRA became useful for memory. Across seeds 42, 43, and 44:

| Seed | BF16 eval | QLoRA eval | Eval delta | Memory delta | Throughput delta |
|---:|---:|---:|---:|---:|---:|
| 42 | 1.37747 | 1.38524 | +0.564% | -23.32% | -19.95% |
| 43 | 1.36653 | 1.37427 | +0.566% | -23.32% | -20.18% |
| 44 | 1.34233 | 1.35478 | +0.927% | -23.32% | -19.98% |

QLoRA 4-bit NF4 is therefore a robust memory-capacity trade-off at 7B on the lab RTX 3090. It is not a throughput win.

The bitsandbytes 8-bit path emitted warnings that fp32/bf16 inputs are cast to fp16 inside `MatMul8bitLt`. We therefore describe that result as int8-weight training with fp16 activation matmul behavior, not pure bf16 compute.

### 4.5 7B Calibration Transfer

The 0.5B absolute thresholds did not transfer directly to 7B. In a targeted 14-module 7B panel, Stage 1 assigned all projections to bf16 because outlier and relative-MSE values were larger at scale. However, rank and perturbation structure did transfer.

Across seeds, the most sensitive paths were:

| Module | Mean abs loss delta | Readout |
|---|---:|---|
| `layers.4.post_attention_layernorm` | 0.2416 | very sensitive |
| `layers.3.mlp.down_proj` | 0.0464 | sensitive |
| `layers.24.mlp.down_proj` | 0.0071 | moderate / borderline |
| `lm_head` | 0.0070 | moderate / borderline |

Four modules were consistently low-delta:

| Module | Mean abs loss delta | Max abs loss delta |
|---|---:|---:|
| `layers.26.mlp.gate_proj` | 0.0014 | 0.0034 |
| `layers.26.mlp.up_proj` | 0.0023 | 0.0044 |
| `layers.27.mlp.gate_proj` | 0.0021 | 0.0034 |
| `layers.26.self_attn.o_proj` | 0.0035 | 0.0046 |

Projection-only outlier score correlated with absolute perturbation delta at approximately 0.78. Int8 relative MSE was not predictive in this panel.

### 4.6 7B Rank-Selected Fake-Int8 Training

We froze a conservative 7B fake-int8 policy using the four consistently low-delta modules:

- `layers.26.mlp.gate_proj`
- `layers.26.mlp.up_proj`
- `layers.27.mlp.gate_proj`
- `layers.26.self_attn.o_proj`

Against matched bf16 controls, the policy preserved LoRA quality:

| Seed | BF16 eval | H6.4 eval | Eval delta | Instability |
|---:|---:|---:|---:|---|
| 42 | 1.37747 | 1.37969 | +0.161% | none |
| 43 | 1.36653 | 1.36958 | +0.223% | none |
| 44 | 1.34233 | 1.34370 | +0.102% | none |

Mean eval degradation was +0.162%, with zero loss spikes and zero NaN/Inf events. This supports calibration-to-training transfer at 7B for a small conservative selected module set.

### 4.7 Backend-Aware Selective Rescue

We next tested whether the same precision-assignment idea can be expressed through a real low-bit backend. On Llama-3.1-8B, the H8 policy starts from QLoRA/NF4 and reloads selected high-risk projection modules in bf16 before LoRA wrapping.

Across matched 500-step RTX 3090 seeds 42, 43, and 44:

| Policy | Mean eval delta vs bf16 | Mean peak-memory delta vs bf16 | Mean throughput delta vs bf16 | Instability |
|---|---:|---:|---:|---|
| QLoRA/NF4 | +0.798% | -26.70% | -19.70% | none |
| QLoRA/NF4 + bf16 projection rescue | +0.682% | -25.28% | -19.17% | none |

Selective rescue improves final eval loss over blanket QLoRA on every seed by about 0.0016 absolute on average, while preserving most of the QLoRA memory saving. It gives up about 0.286 GiB peak memory relative to blanket QLoRA and does not remove the low-bit throughput penalty.

We then instantiated the H10 assignment table from these matched H8 summaries. The generated table reports blanket QLoRA quality risk `0.00798336`, selective-rescue quality risk `0.00681742`, memory deltas of `-5.391 GiB` and `-5.105 GiB` versus bf16, and zero instability risk. Under a 1% quality-risk gate and required memory saving, the solver selects QLoRA/NF4 plus bf16 projection rescue.

A selector-aware planning table further compares possible top-k rescue selectors. On Llama-3.1-8B at `k=4`, target perturbation ranking captures all three unsafe projection candidates, while activation outliers, INT8 MSE, role priors, and cross-model predictors each capture only one of three. This argues for target perturbation-guided rescue as the current defensible method, with learned cross-model predictors treated as extensions rather than replacements.

### 4.8 Inference-Side PTQ Assignment

The active H10 contribution follows the original HAQ deployment setting: post-training quantization for inference. We therefore built an inference action table from H9 vLLM benchmark and prompt-logprob artifacts, keeping each policy-workload-backend row explicit and solving for Pareto-feasible policies under a strict quality gate.

The final matched table uses Llama-3.1-8B-Instruct bf16/fp16 baselines and matched AWQ-Marlin and GPTQ-Marlin artifacts. Under the locked constraint `predicted_quality_risk <= 0.01`, the solver selects `llama31_8b_instruct_gptq_marlin_artifact` for all three matched Instruct workloads:

| Workload | Prompt NLL delta vs bf16 | Latency delta vs bf16 | Output tok/s delta vs bf16 |
|---|---:|---:|---:|
| decode_heavy | +0.773771% | -62.721734% | +168.254866% |
| mixed | +0.773771% | -62.445208% | +166.278850% |
| prefill_heavy | +0.773771% | -60.494807% | +153.139001% |

AWQ-Marlin is backend-feasible and similarly fast, but its prompt-NLL delta is +2.851377%, so it fails the strict gate and is only accepted in a relaxed 3% sensitivity analysis. This gives the first strict positive H10 result: a backend-real inference PTQ policy selected by the action-table solver passes the locked quality gate and improves the deployment frontier.

## 5. Discussion

The central result is not that fake-int8 hooks make training faster. They do not. The central result is that module sensitivity is measurable before training, and those measurements identify modules whose precision can be changed or rescued without breaking LoRA updates.

The hardware-aware extension turns this into a PEFT adaptation of the HAQ principle. The durable HAQ idea is not reinforcement learning over CNN layer bitwidths; it is hardware-aware precision assignment under measured quality and resource constraints. In this project, the corresponding object is a table where each row is a candidate module-format-backend action:

```text
model_size, seed, module_name, layer_idx, module_role,
activation_outlier_score, int8_rel_mse, perturbation_delta,
backend, candidate_action, backend_feasible,
measured_memory_delta, measured_throughput_delta,
selected_by_policy, training_safe_label
```

A simple predictor can then estimate:

```text
risk(module_i, format_f, backend_b) = expected quality degradation
```

and a budgeted optimizer can choose assignments under a memory, throughput, or quality constraint. This turns the high-school-style permutation/combination space into a constrained prediction and optimization problem. The PEFT-specific constraint is backend feasibility: an assignment is only useful if the LoRA/QLoRA stack can actually express it without destroying the intended memory or throughput benefit.

The selector experiments also clarify what should not be claimed. Cross-model learned predictors are not yet strong enough to replace target perturbation checks: on Llama-3.1-8B, the cheap selectors and learned transfer predictors recover only one of three unsafe projection candidates at `k=4`. The main method should therefore be described as measured target calibration and perturbation followed by backend-aware assignment. Learned predictors are promising only as future amortization once more labeled models exist.

For H10, the project-aligned version of the same idea is inference-side PTQ rather than PEFT. The final GPTQ-Marlin result shows why the table needs backend feasibility, quality, latency, throughput, memory, and workload identity in the same object: AWQ-Marlin is fast but fails the strict prompt-NLL gate, while GPTQ-Marlin passes the gate and gives large latency and throughput wins. The result should be reported as a Pareto-frontier deployment claim, not as a generic statement that all quantized artifacts are safe.

## 6. Limitations

The current selective low-precision path is fake-int8 output quantization implemented with Python hooks. It is useful for sensitivity testing but not a resource-saving kernel. The backend-real selective-rescue path addresses memory, but it is still a narrow QLoRA/NF4 plus bf16 projection-rescue prototype.

The experiments use Qwen2.5 models and Alpaca-style LoRA fine-tuning. The results may not transfer unchanged to other model families, downstream tasks, sequence lengths, learning rates, or full fine-tuning.

The 7B selective calibration panel is targeted, not all-module. The conservative 7B policy is intentionally small. Wider 7B policies remain future work.

Validation NLL is the primary metric. The project does not yet include a full downstream benchmark suite.

QLoRA and selective fake-int8 are separate interventions for most Qwen experiments. The Llama selective-rescue result shows a first calibration-guided improvement over blanket QLoRA, but it is modest and should be treated as a memory-quality trade-off, not a throughput win.

The H10 selector-aware table is partly a planning artifact. It compares selector policies by scaling measured H8 top-4 rescue recovery by unsafe recall, so it should guide which policy deserves validation rather than substitute for a fresh training run.

The active H10 inference result is currently limited to matched Llama-3.1-8B-Instruct vLLM workloads and prompt-NLL quality scoring. It should be replicated on another model or artifact family and checked with downstream task metrics before claiming broad PTQ generality. It also uses whole-artifact GPTQ/AWQ choices rather than true layer-wise mixed precision; a backend-supported layer/group assignment path remains future work.

## 7. Related Work

Mixed precision training was formalized by Micikevicius et al. as a recipe combining low-precision tensors with higher-precision safeguards such as master weights and loss scaling. BF16 training work motivates bf16 as a strong baseline because it preserves FP32-like exponent range. FP8 and FP4 training papers further show that different tensors and operations require different numerical treatment.

LoRA and QLoRA define the low-resource adaptation setting closest to this project. QLoRA is especially relevant because it demonstrates that quantized base models can support effective parameter-efficient fine-tuning. Our work differs by studying module-level sensitivity and selective precision assignment rather than applying a single global quantized backend.

LLM.int8(), SmoothQuant, and ZeroQuant motivate activation-aware and module-aware quantization for Transformers. These works support our use of activation outlier statistics as a signal, although they primarily target inference or post-training quantization. HAQ, HAWQ, and HAWQ-V3 frame mixed precision as sensitivity-aware or hardware-aware constrained optimization. Recent adaptive and subbyte training work, including convergence-aware operator-wise mixed precision, FP4 LLM training, SNIP-style adaptive subbyte training, and attention quantization-aware training, further motivates operation-wise precision assignment during training.

This project contributes a small but concrete LoRA-focused empirical result: cheap calibration and perturbation probes can select module precision assignments that preserve training quality across seeds and scale to a conservative 7B panel. It also gives a first backend-aware PEFT instantiation of HAQ-style assignment through QLoRA/NF4 plus selected bf16 projection rescue, and a project-aligned inference PTQ instantiation through matched vLLM GPTQ-Marlin action-table selection.

## 8. Conclusion

Precision assignment in Transformer fine-tuning is a structured combinatorial problem. This project shows that cheap calibration signals and one-module perturbation probes can reduce that problem to a measured ranking over candidate modules. At 0.5B, calibration-selected fake-int8 policies preserve bf16 validation quality across seeds and can be widened to 24 MLP gate/up modules. At 7B, fixed thresholds from 0.5B fail, but rank/perturbation selection transfers: a conservative four-module fake-int8 policy preserves LoRA quality across seeds with mean eval degradation of only +0.162%.

The current evidence supports calibration-guided sensitivity ranking and quality-preserving policy selection. It does not support a resource-saving claim for the fake-int8 implementation. Hardware-backed QLoRA separately gives a robust 7B memory-capacity trade-off, and selective bf16 projection rescue from QLoRA/NF4 improves the quality side of that trade-off on Llama-3.1-8B while retaining most memory savings. In the active H10 inference setting, matched GPTQ-Marlin serving passes the strict 1% prompt-NLL gate and substantially improves latency and output throughput versus bf16 on Llama-3.1-8B-Instruct.

The next step is to test whether the H10 inference result replicates beyond one artifact and model family, add task-level quality checks beyond prompt NLL, and pursue a backend-supported path from whole-artifact PTQ choices toward true layer/group mixed precision.

## Citation Notes

The related-work papers are already summarized under `literature/` and listed in `research-state.yaml`. Before LaTeX submission, BibTeX entries should be fetched and verified programmatically rather than generated from memory.
