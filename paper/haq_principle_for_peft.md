# Archived Note: Adapting the HAQ Principle to PEFT

Draft date: 2026-05-26

## Scope Status

This note is retained as a historical PEFT-side formulation. It is no longer the
active H10 direction. The active project-plan-aligned H10 scope is
inference-side mixed-precision post-training quantization:

```text
experiments/h10-inference-ptq-assignment/protocol.md
```

The PEFT framing below may still be useful as supporting evidence for
calibration and perturbation sensitivity, but it should not be presented as the
main HAQ-aligned contribution. Original HAQ is inference-oriented, and the
project plan's "post-training quantization" target should be interpreted as PTQ
for inference deployment.

## Core Reframing

HAQ's durable idea is not "use reinforcement learning for CNN bitwidths." The durable idea is:

> Precision assignment is a hardware-constrained policy problem, and the policy should be chosen with feedback from the target backend rather than by global dtype defaults or proxy FLOP counts.

For PEFT, the analogous problem is not per-layer CNN bitwidth search. It is:

> Learn or estimate Transformer module precision risk from cheap probes, then solve a backend-aware precision assignment problem over the precision states that the PEFT stack can actually implement.

This makes the project's H6/H7/H8 trajectory a PEFT adaptation of the HAQ principle:

1. Replace hand-picked precision islands with measured module risk.
2. Replace exhaustive combinatorial search with a predictor or cheap probe table.
3. Replace abstract bitwidth choices with backend-real actions such as QLoRA/NF4, bf16 rescue, bitsandbytes 8-bit, adapter dtype, or optimizer-state precision.
4. Choose a frozen policy before training and validate it against matched bf16 and blanket low-bit baselines.

In short:

```text
HAQ:
  state = CNN layer index and resource budget
  action = choose layer bitwidth
  feedback = hardware latency/energy plus accuracy
  optimizer = reinforcement learning policy search

HAQ-for-PEFT:
  state = Transformer module, role, calibration signals, backend, hardware
  action = choose feasible PEFT precision action
  feedback = predicted precision risk plus measured backend memory/throughput
  optimizer = constrained assignment over module/action rows
```

The analogy is deliberately principle-level. The original method's RL controller is not the important part for PEFT. The important part is closing the loop between precision choices, accuracy risk, and target hardware feedback.

## Why PEFT Changes the Search Space

In CNN inference, HAQ can treat each layer's activation/weight bitwidth as the main decision. In LoRA or QLoRA fine-tuning, the decision sites and constraints are different:

- The base Transformer weights are usually frozen.
- Trainable parameters are mostly adapter matrices and optimizer states.
- Quantized base weights may be packed by a backend, so arbitrary casting after load may not be possible.
- Activations, adapter updates, gradients, loss computation, norms, and projection outputs can have different precision sensitivity.
- Resource savings only count if the selected action maps to a real backend path.

Therefore the PEFT policy should assign an action to each module or module group:

```text
a_i in feasible_actions(module_i, backend, hardware)
```

Example feasible actions:

```text
bf16_lora_default
qlora_nf4_default
bnb_int8_lora
qlora_nf4_with_bf16_rescue
qlora_nf4_with_fp32_rescue
adapter_bf16
adapter_fp32
optimizer_8bit
```

The key is feasibility. A policy that says "make this packed NF4 module bf16" is not an assignment until the implementation can actually reload or replace that module while preserving the rest of the low-bit backend.

## Risk Model

Each candidate row should be a `(model, module, candidate_action, backend)` tuple:

```text
model_name
model_size_b
module_name
layer_idx
normalized_depth
module_role
candidate_action
backend
hardware_label
activation_outlier_score
input_outlier_score
output_outlier_score
quantization_error
one_module_perturbation_delta
mini_update_divergence
instability_flags
backend_feasible
measured_memory_delta
measured_throughput_delta
predicted_quality_risk
```

The target is not simply "safe or unsafe module." It is:

```text
risk(module_i, action_f, backend_b) =
  expected quality/stability degradation if action_f is used for module_i on backend_b
```

The existing H7 predictor is the first version of this idea for fake-int8 output perturbation. A stronger HAQ-for-PEFT version should extend it from:

```text
risk(module)
```

to:

```text
risk(module, format/action, backend)
```

This matters because the same module can be safe under one action and unsafe or irrelevant under another. For example, a late MLP gate projection may be low risk for fake-int8 output perturbation, medium risk for NF4 base-weight storage, and impossible to express under a given vLLM launch policy. The risk label must therefore attach to the action and backend, not only to the module.

The minimum viable label set is:

```text
safe_for_training = final_eval_delta <= epsilon and no extra instability
quality_delta = final_eval_loss(policy) - final_eval_loss(matched_baseline)
memory_delta = peak_memory(policy) - peak_memory(matched_baseline)
throughput_delta = train_or_inference_throughput(policy) - throughput(matched_baseline)
backend_feasible = can the stack actually express the action?
```

For the first assignment optimizer, predicted risk can be conservative:

```text
predicted_quality_risk(row) =
  calibrated_predictor(row.features)
  or rank_normalized(abs(one_module_perturbation_delta))
  or max over seed-level perturbation deltas
```

The first paper claim should not require a sophisticated learner. A cheap, auditable estimator is preferable if it produces an implementable policy.

## Cheap Probes

The PEFT search should avoid full training for every precision recipe. Useful cheap probes are:

- module inventory and backend feasibility checks,
- activation outlier statistics,
- input/output fake-quantization error,
- one-module perturbation loss deltas,
- short calibration loss under candidate low-bit actions,
- optional mini-update probes that compare adapter-gradient or update drift over a few batches.

The H6/H7 evidence suggests that activation outliers and one-module perturbation deltas are especially useful for projection modules. Output quantization probes are weaker for norms and logits because they do not test internal reduction or loss-computation precision.

## Assignment Objective

Once each feasible action has a predicted risk and measured or estimated backend cost, policy selection becomes a constrained optimization problem:

```text
choose policy a

minimize      measured_or_predicted_resource_cost(a; hardware)
subject to    predicted_quality_risk(a) <= epsilon
              predicted_instability_risk(a) <= tau
              a_i in feasible_actions(i, backend, hardware)
              policy is frozen before final training validation
```

Equivalently, for selective rescue from a low-bit baseline:

```text
maximize      quality_recovery(a) - lambda * rescue_cost(a)
subject to    memory_saving(a) >= target_saving
              final_eval_delta_predicted(a) <= epsilon
```

This is the PEFT counterpart to HAQ's hardware-aware bitwidth assignment. The solver can start simple:

- greedy ranking by risk saved per GiB added,
- top-k rescue under a memory budget,
- small knapsack over module groups,
- integer programming once the feasible action table is reliable.

Because single-module risks are not guaranteed to add linearly, the first policy space should use groups such as:

- norm/logit paths,
- early high-risk MLP down projections,
- selected attention projections,
- selected late MLP gate/up projections,
- remaining projection modules.

The assignment should also include a baseline action per group. This prevents the optimizer from silently omitting modules and makes the policy executable:

```text
for every group g:
  choose exactly one action from feasible_actions(g)
```

The first solver can be intentionally small:

1. Enumerate all actions when the grouped action space has at most a few thousand combinations.
2. Otherwise run greedy rescue by benefit per memory cost.
3. Keep integer programming as a later engineering step, not a prerequisite for the research claim.

## Backend-Aware Policy Pattern

The strongest PEFT pattern is not "demote a few bf16 modules." That tends to preserve quality but rarely saves resources. The stronger systems pattern is:

1. Start from a hardware-backed low-bit baseline such as QLoRA/NF4.
2. Use probes or a predictor to identify modules where low-bit precision has high quality risk.
3. Rescue only those modules to bf16/fp32 if the backend can express it.
4. Keep the rest of the model on the low-bit path so the memory benefit remains real.

This is exactly what H8 demonstrated in first-pass form: QLoRA/NF4 plus selected bf16 projection rescue improved final eval loss over blanket QLoRA on every matched Llama-3.1-8B seed while preserving most of the memory saving. The effect is modest, but it validates the PEFT version of HAQ's principle: combine sensitivity estimates with backend cost, then choose an implementable mixed-precision policy.

The design consequence is important for the paper:

- H6/H7 prove the cheap-risk-estimation side: probes can identify modules that tolerate or reject reduced precision.
- H8 proves the backend-expression side: a selected precision policy can be implemented as low-bit baseline plus bf16 projection rescue.
- H10 should join them into one table-and-solver artifact: risk rows, cost rows, feasibility rows, selected policy.

That is the clean adaptation of HAQ to PEFT.

## Proposed Algorithm

```text
Input:
  model M
  PEFT method P
  calibration data D_cal
  backend set B
  hardware target H
  quality budget epsilon
  resource objective C

1. Build module inventory I from M and P.
2. For each module/group i in I:
     collect cheap probe features x_i on D_cal.
3. For each feasible backend action a in A(i, B, H):
     estimate risk r_i,a from probes, perturbation deltas, or a trained predictor.
     measure or estimate cost c_i,a from backend probes.
     mark backend feasibility f_i,a.
4. Solve:
     choose one action a_i for each group i
     minimizing total cost
     subject to risk <= epsilon and feasibility = true.
5. Freeze the selected policy before fine-tuning.
6. Validate against matched bf16 and blanket low-bit baselines.
```

The assignment object is the main artifact, not the solver family. HAQ used RL because the CNN hardware-aware bitwidth search was large. PEFT can start with grouped enumeration or greedy knapsack because the first implementable backend actions are much coarser.

## Paper-Level Claim

A clean formulation for the paper is:

> We adapt the HAQ principle from CNN inference bitwidth search to PEFT fine-tuning. Rather than using reinforcement learning over CNN layer bitwidths, we estimate Transformer module precision risk with cheap calibration and perturbation probes, then choose a backend-feasible precision assignment under a quality and resource budget. This turns PEFT precision selection from a global recipe choice, such as bf16 versus QLoRA, into a constrained module/action assignment problem.

The honest boundary is equally important:

> The current evidence supports cheap risk estimation and a first backend-aware selective-rescue policy. It does not yet show a general-purpose optimizer over all Transformer modules, formats, and backends.

## Minimal Next Artifact

The next implementation artifact is a single policy table and a small solver:

```text
experiments/h10-haq-peft-assignment/
  protocol.md
  code/
    build_action_table.py
    measure_backend_costs.py
    solve_precision_assignment.py
  results/
    action_table.csv
    selected_policy.json
    backend_costs.json
```

The first version can reuse H7 risk features and H8 backend feasibility/cost measurements. Its job is not to discover a new result immediately; its job is to make the HAQ-for-PEFT abstraction executable.

The first H10 policy should be modest and confirmatory:

```text
baseline action: blanket QLoRA/NF4
candidate rescue action: bf16 projection rescue
risk signal: H7/H8 high-risk projection ranking
cost signal: H8 measured +0.286 GiB for top projection rescue, with retained memory saving versus bf16
quality gate: <= 1% eval-loss degradation against matched bf16
resource gate: retain at least 20% peak-memory saving against matched bf16
```

This avoids overclaiming. It says: given cheap risk estimates and measured backend costs, can the assignment procedure recover the H8-style memory-quality trade-off as the selected policy?

## Citation Anchors

- HAQ: Hardware-Aware Automated Quantization with Mixed Precision, Wang et al., CVPR 2019, https://arxiv.org/abs/1811.08886
- LoRA: Low-Rank Adaptation of Large Language Models, Hu et al., 2021, https://arxiv.org/abs/2106.09685
- QLoRA: Efficient Finetuning of Quantized LLMs, Dettmers et al., NeurIPS 2023, https://arxiv.org/abs/2305.14314
- HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision, Dong et al., ICCV 2019, https://arxiv.org/abs/1905.03696
