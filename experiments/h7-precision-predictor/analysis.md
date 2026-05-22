# H7 Analysis

## 2026-05-20 First Predictor Artifact

Implemented the first H7 dataset and predictor pipeline over existing H6 artifacts.

Artifacts:

- `results/precision_dataset.csv`
- `results/predictor_metrics.json`
- `results/predictions.csv`
- `results/selected_policy.json`
- `results/selected_modules.txt`

The dataset builder produced 696 module rows, including 72 perturbation-labeled rows. The labeled rows contain three seeds each for Qwen2.5-0.5B and a targeted Qwen2.5-7B panel:

| model | labeled rows per seed | total labeled rows |
|---|---:|---:|
| Qwen2.5-0.5B | 10 | 30 |
| Qwen2.5-7B | 14 | 42 |

At the locked safe threshold `abs_perturbation_delta <= 0.005`, there are 44 safe rows and 28 unsafe rows.

### Cross-Scale Result

The main H7 check trains on 0.5B perturbation rows and tests on the held-out 7B targeted panel.

| method | Spearman with abs delta | AUROC unsafe | precision@4 lowest risk | precision@8 lowest risk |
|---|---:|---:|---:|---:|
| Ridge predictor | `0.536` | `0.776` | `1.000` | `0.875` |
| Logistic safe probability | `0.493` | `0.760` | `1.000` | `0.875` |
| Outlier-only rank | `0.484` | `0.750` | `0.750` | `0.875` |
| INT8-MSE-only rank | `0.089` | `0.535` | `0.750` | `0.750` |
| Fixed thresholds | n/a | n/a | `0.750` | `0.875` |

Interpretation: the learned predictor modestly improves over pure outlier ranking on the 0.5B-to-7B transfer panel, especially for top-4 safe selection. This is the first evidence that combining module role, depth, outlier score, and quantization-error features can improve precision-risk ranking beyond a single signal.

The improvement is small. Outlier ranking remains a strong baseline and should not be dismissed.

### Leave-One-Seed-Out Result

Leave-one-seed-out is mixed:

| held-out seed | best learned signal | outlier-only comparison |
|---:|---|---|
| 42 | logistic Spearman `0.504`, AUROC `0.758` | outlier Spearman `0.483`, AUROC `0.781` |
| 43 | ridge Spearman `0.611`, AUROC `0.807` | outlier Spearman `0.659`, AUROC `0.896` |
| 44 | logistic Spearman `0.759`, AUROC `0.832` | outlier Spearman `0.752`, AUROC `0.857` |

Interpretation: the predictor is not uniformly better than outlier ranking within-scale. The current labeled dataset is small, so H7 should be framed as an implementation artifact and early signal, not as a completed superiority claim.

### Selected Cross-Scale 7B Policy

Using Ridge predicted risk on the held-out 7B projection rows, the top-4 selected modules are:

- `base_model.model.model.layers.26.self_attn.o_proj`
- `base_model.model.model.layers.27.mlp.gate_proj`
- `base_model.model.model.layers.26.self_attn.q_proj`
- `base_model.model.model.layers.27.mlp.up_proj`

This differs from the conservative H6.4 policy:

- `base_model.model.model.layers.26.mlp.gate_proj`
- `base_model.model.model.layers.26.mlp.up_proj`
- `base_model.model.model.layers.27.mlp.gate_proj`
- `base_model.model.model.layers.26.self_attn.o_proj`

The predictor-selected set is more aggressive because it includes `layers.26.self_attn.q_proj` and `layers.27.mlp.up_proj`, both of which have low mean perturbation deltas but crossed the strict per-seed `0.005` safe threshold in one seed. Therefore, this selected policy should not be treated as validated for training yet.

When restricted to MLP `gate_proj` / `up_proj` modules, the selector emits:

- `base_model.model.model.layers.27.mlp.gate_proj`
- `base_model.model.model.layers.27.mlp.up_proj`
- `base_model.model.model.layers.26.mlp.gate_proj`
- `base_model.model.model.layers.26.mlp.up_proj`

This is close to the H6.4 conservative policy but swaps in `layers.27.mlp.up_proj` instead of the attention `o_proj`. It is a sensible next exploratory policy if we want to test a predictor-selected 7B module set.

### Current Conclusion

H7 now exists as a working implementation. The strongest honest claim is:

> A simple predictor trained on 0.5B calibration/perturbation rows modestly improves 7B top-4 safe-module selection over pure outlier ranking, but outlier ranking remains highly competitive and the labeled dataset is small.

The next useful step is to either:

1. run the predictor-selected 7B top-4 policy as a new exploratory training comparison, or
2. improve the predictor by training on aggregated module-level rows rather than per-seed rows and adding a conservative max-delta objective.

For paper framing, H7 is currently a promising extension, not yet part of the main confirmed contribution.

## 2026-05-21 Llama-3.1-8B Conservative Policy Training

The first matched 500-step Llama-3.1-8B training validation is complete on `a100-colab`. This tested the conservative two-module policy selected from the three-seed Llama perturbation probes:

- `base_model.model.model.layers.30.mlp.gate_proj`
- `base_model.model.model.layers.30.mlp.up_proj`

Run setup:

| field | value |
|---|---|
| model | `meta-llama/Llama-3.1-8B` |
| hardware | `NVIDIA A100-SXM4-40GB`, `HARDWARE_LABEL=a100-colab` |
| seed | `42` |
| max steps | `500` |
| sequence length | `512` |
| effective batch size | `16` |
| learning rate | `2e-4` |
| eval max batches | `100` |

Matched result:

| metric | bf16 | conservative fake-int8 MLP2 | delta |
|---|---:|---:|---:|
| final eval loss | `1.389425` | `1.390033` | `+0.000608` (`+0.044%`) |
| final train loss | `1.234381` | `1.230977` | `-0.003405` |
| max grad norm | `5.6296` | `5.6494` | `+0.0198` |
| loss spikes | `0` | `0` | no change |
| NaN/Inf events | `0` | `0` | no change |
| peak CUDA memory GiB | `20.1955` | `20.1955` | no change |
| train tok/s excl. first | `278.56` | `277.00` | `-0.56%` |

Interpretation: this is a positive first training validation for cross-family transfer. The Llama-3.1-8B conservative fake-int8 policy preserved bf16 validation quality and stability on seed 42, with degradation far inside the locked 1% gate. As with the Qwen fake-int8 policies, this is not a memory-saving result because the implementation is a Python-level fake-quant output hook.

This seed-42 result justified replication on seeds 43 and 44.

## 2026-05-22 Llama-3.1-8B Conservative Policy Replication

The conservative two-module fake-int8 MLP policy is now replicated across seeds 42, 43, and 44 with matched bf16 controls.

| seed | bf16 eval | H7 eval | abs delta | rel delta | spikes total | NaN/Inf total | memory delta | tok/s delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | `1.389425` | `1.390033` | `+0.000608` | `+0.044%` | 0 | 0 | `+0.000000 GiB` | `-0.56%` |
| 43 | `1.364863` | `1.366315` | `+0.001452` | `+0.106%` | 0 | 0 | `+0.000000 GiB` | `+1.86%` |
| 44 | `1.361036` | `1.361887` | `+0.000852` | `+0.063%` | 0 | 0 | `+0.000000 GiB` | `+0.07%` |

Mean paired eval degradation is `+0.000970` absolute, or `+0.071%` relative. The worst seed is `+0.106%`, far inside the locked 1% quality gate. Both policies had zero loss spikes and zero NaN/Inf events in all paired runs.

Interpretation: H7 now has replicated cross-family calibration-to-training transfer for a conservative Llama-3.1-8B selected module set. This strengthens the story beyond Qwen: a short perturbation-calibrated precision check can identify low-risk projection modules that remain harmless during actual LoRA updates across model families.

The resource interpretation is unchanged. Peak memory is identical because fake-int8 output hooks do not change model storage, optimizer states, or activation allocation in a hardware-backed way. Throughput deltas are small and noisy. The claim is quality-preserving sensitivity assignment, not resource savings.

Next step: run a cautious wider Llama policy. The best next policy should add only the two stable attention modules from the perturbation probe:

- `base_model.model.model.layers.30.self_attn.q_proj`
- `base_model.model.model.layers.30.self_attn.o_proj`

to the validated conservative MLP pair:

- `base_model.model.model.layers.30.mlp.gate_proj`
- `base_model.model.model.layers.30.mlp.up_proj`

Avoid adding `layers.31.mlp.up_proj` or `layers.31.mlp.gate_proj` yet, because they were not stable under the strict perturbation safety rule across seeds.

## 2026-05-22 Llama-3.1-8B Cautious Attention+MLP Policy

The cautious four-module Llama policy completed for seed 42. The run output landed at `experiments/h7-precision-predictor/results/summary.json` and `metrics.jsonl` rather than the intended named subdirectory, but the summary metadata confirms the policy:

- `base_model.model.model.layers.30.self_attn.q_proj`
- `base_model.model.model.layers.30.self_attn.o_proj`
- `base_model.model.model.layers.30.mlp.gate_proj`
- `base_model.model.model.layers.30.mlp.up_proj`

Matched seed-42 comparison:

| metric | bf16 | conservative MLP2 | cautious attention+MLP4 |
|---|---:|---:|---:|
| final eval loss | `1.389425` | `1.390033` | `1.390232` |
| delta vs bf16 | - | `+0.044%` | `+0.058%` |
| loss spikes | `0` | `0` | `0` |
| NaN/Inf events | `0` | `0` | `0` |
| peak CUDA memory GiB | `20.1955` | `20.1955` | `20.1955` |
| train tok/s excl. first | `278.56` | `277.00` | `272.59` |

Interpretation: the cautious four-module policy is safe on seed 42. It remains far inside the 1% quality gate and has no instability. The eval degradation is slightly larger than the conservative two-module policy (`+0.058%` vs `+0.044%` relative to bf16), but the difference is tiny. Throughput is about `2.14%` lower than bf16 under Python fake-int8 hooks, which should not be treated as hardware-realistic performance.

Next step: replicate this cautious four-module policy on seeds 43 and 44. If it holds, then H7 has a stronger cross-family result: Llama can tolerate both a small selected MLP pair and selected attention projections during LoRA updates.
