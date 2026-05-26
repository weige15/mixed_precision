# H10 Protocol: Backend-Aware PEFT Precision-Risk Prediction

## Question

Can a short calibration and perturbation probe predict which Transformer
projection modules should be rescued to high precision from a low-bit PEFT
baseline, improving the quality-memory trade-off over blanket QLoRA?

## Motivation

H6 and H7 show that module-level calibration signals can rank precision
sensitivity during LoRA training. H8 shows that this ranking can be expressed
through a hardware-real QLoRA/NF4 baseline plus selective bf16 projection
rescue. H10 makes that connection explicit: the main research target is not
another global dtype comparison, but a predictor for
`risk(module, format, backend, phase=peft_training)`.

This is HAQ-inspired, not a direct HAQ reproduction. HAQ searches CNN inference
bitwidths under hardware constraints. H10 searches PEFT precision/rescue
policies under fine-tuning quality, stability, memory, and backend constraints.

## Policy Space

The first H10 backend-real policy space is:

```text
base policy: QLoRA/NF4
action: rescue selected quantized projection modules to bf16 before LoRA wrapping
budget: top-k rescued projection modules
target phase: PEFT training
```

The first implementation keeps norms and logits out of the rescue search because
H8 feasibility checks showed they are already outside the bitsandbytes 4-bit
projection modules in the current PEFT path.

## Comparators

For a fixed rescue budget `k`, compare:

- blanket bf16 LoRA
- blanket QLoRA/NF4
- random projection rescue
- role-prior projection rescue
- activation-outlier rescue
- cross-model learned predictor rescue
- oracle perturbation rescue as an upper-bound diagnostic only

The key claim is supported only if calibration-guided rescue beats random and
simple role-only rescue at the same rescue budget while preserving most QLoRA
memory savings.

## Labels

Seed-level perturbation rows are aggregated into module-level labels:

```text
mean_abs_delta
max_abs_delta
safe_rate
safe_all_seeds = max_abs_delta <= safe_threshold
safe_majority = safe_rate >= 0.5
```

The conservative H10 target is `max_abs_delta`, because a policy that fails one
seed is risky for a frozen precision assignment.

## Decision Rules

H10 is supported if a learned or calibration-guided rescue selector:

- improves final eval loss over blanket QLoRA on most matched seeds,
- stays within 1% final eval-loss degradation versus matched bf16,
- preserves most of the QLoRA peak-memory saving,
- does not increase loss spikes or NaN/Inf events,
- beats random and role-prior rescue at the same rescue budget.

H10 is partially supported if the predictor does not beat activation-outlier
ranking but the calibration-guided policy still improves the QLoRA
memory-quality trade-off. In that case, the honest claim is that cheap
calibration signals are sufficient and learned prediction is not yet justified.

