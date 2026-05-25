# H8 Analysis

## 2026-05-24 Bootstrap

H8 starts as a hardware-aware extension of H7. The goal is to move from a risk-only precision predictor to a policy optimizer that reasons about both:

- predicted quality/stability risk, and
- measured backend cost on the target hardware.

The immediate design choice is to prioritize **selective rescue from a hardware-backed low-bit baseline** over selective demotion from bf16. This follows both the local evidence and related work:

- H6/H7 fake-int8 policies preserve quality but do not save memory.
- H6.3 QLoRA on Qwen2.5-7B saves memory but degrades quality slightly and is slower.
- HAQ/DNAS/HAWQ-style work frames mixed precision as constrained search.
- AWQ/OWQ/SpQR/SqueezeLLM/QServe-style LLM systems often start from low precision and protect sensitive parts.

Implemented first planning script:

- `code/build_h8_policy_candidates.py`

The script reads the H7 precision dataset, aggregates perturbation-labeled module risks, and emits candidate selective-rescue policies:

- `h8_rescue_norm_logits`
- `h8_rescue_norm_logits_highrisk_down`
- `h8_rescue_projection_top{k}`

These are planning artifacts only. They do not prove that the backend can express true per-module rescue while retaining QLoRA memory savings.

Next implementation task: run the planner, inspect candidate modules, then check whether the current runner/backend can express any of the rescue policies without destroying the QLoRA memory benefit.

## 2026-05-24 Qwen/Llama Alignment

Regenerated `results/h8_policy_candidates.json` from the combined H7 dataset:

- `experiments/h7-precision-predictor/results/precision_dataset_with_llama31_8b.csv`

The candidate file now tracks both:

- `Qwen/Qwen2.5-7B`
- `meta-llama/Llama-3.1-8B`

Both models have three-seed perturbation labels for 14 modules and are marked `candidate_ready`. The policy sets are still planning artifacts with `backend_feasibility: unverified`; the next H8 step is a backend feasibility check for whether QLoRA/NF4 can selectively rescue the proposed modules to bf16/fp32 without losing the memory advantage.

Added `code/inspect_h8_backend_feasibility.py` for this check. It mirrors the QLoRA load path, applies PEFT LoRA wrapping by default, resolves the candidate rescue targets, and reports whether each target is missing, already non-quantized, or backed by a bitsandbytes quantized linear module. The local environment has the packages installed but no available CUDA device, so the real backend probe must run on the target RTX 3090 host.

The first RTX 3090 feasibility probes completed for `h8_rescue_norm_logits`:

- Qwen/Qwen2.5-7B: 3 targets were `already_non_quantized`; `lm_head` was `non_quantized_linear`.
- meta-llama/Llama-3.1-8B: 2 targets were `already_non_quantized`; `lm_head` was `non_quantized_linear`.

Interpretation: norm/logit rescue is not a meaningful H8 selective rescue under the current QLoRA/NF4 backend, because these paths are already outside bitsandbytes 4-bit quantized linear modules. The next feasibility target should be `h8_rescue_projection_top4`, where candidate modules are expected to be PEFT-wrapped bitsandbytes linear layers.

`h8_rescue_projection_top4` feasibility also completed on the RTX 3090:

- Qwen/Qwen2.5-7B: all 4 projection targets are `quantized_target` modules, represented as PEFT `Linear4bit` wrappers over bitsandbytes `Linear4bit` base layers with `torch.uint8` weights.
- meta-llama/Llama-3.1-8B: all 3 projection targets are `quantized_target` modules with the same PEFT/bitsandbytes structure.

Interpretation: projection rescue is the right H8 implementation target. It cannot be implemented by simply casting the existing modules, because the weights are already packed 4-bit/uint8 bitsandbytes parameters. H8 needs either a module replacement path that reloads selected projection weights in bf16/fp32, or a controlled approximation/prototype that measures the resource cost of adding high-precision shadow modules for those targets.

Implemented first real selective-rescue path in `experiments/h1-selective-fp32-norms/code/run_lora_precision.py`:

- new policy: `h8_qlora_nf4_rescue_projection_top4`
- reads `results/h8_policy_candidates.json`
- maps PEFT candidate names such as `base_model.model.model.layers.31.mlp.up_proj` back to pre-PEFT model names such as `model.layers.31.mlp.up_proj`
- loads selected checkpoint tensors from safetensors or PyTorch shards
- replaces the corresponding bitsandbytes 4-bit modules with frozen bf16/fp32 `torch.nn.Linear` modules
- applies LoRA after replacement so rescued modules can still receive adapters
- records `h8_rescued_modules` in setup/training summaries

Added `--setup-only` to verify model loading, replacement, LoRA wrapping, and peak setup memory without starting dataset processing or training. The next empirical step is a setup-only smoke test on the RTX 3090, followed by a 100-step selective-rescue run before spending a full 500-step run.

## 2026-05-26 Close-Out

H8 now has a complete first-pass empirical answer for Llama-3.1-8B LoRA on the lab RTX 3090. The implemented selective-rescue policy starts from QLoRA/NF4, reloads the selected high-risk projection modules in bf16, then applies LoRA. The matched comparison set covers bf16, blanket QLoRA/NF4, and H8 selective rescue for 500 optimizer steps across seeds 42, 43, and 44 under the same `rtx3090-lab` hardware label.

Aggregated 500-step RTX 3090 results:

| Policy | Mean eval delta vs bf16 | Mean peak-memory delta vs bf16 | Mean train-throughput delta vs bf16 | Instability |
|---|---:|---:|---:|---|
| QLoRA/NF4 | +0.798% | -26.697% | -19.698% | 0 spikes, 0 NaN/Inf |
| H8 selective rescue | +0.682% | -25.280% | -19.174% | 0 spikes, 0 NaN/Inf |

Paired seed-level H8 improvement over blanket QLoRA:

| Seed | QLoRA eval loss | H8 eval loss | H8 improvement | Added memory vs QLoRA |
|---:|---:|---:|---:|---:|
| 42 | 1.398847 | 1.397240 | 0.001607 | +0.286 GiB |
| 43 | 1.377077 | 1.375443 | 0.001634 | +0.286 GiB |
| 44 | 1.371851 | 1.370295 | 0.001557 | +0.286 GiB |

Mean H8 improvement over QLoRA is `0.001599` final eval loss. This is small but directionally consistent across all three seeds. H8 gives up about `0.286 GiB` peak memory relative to blanket QLoRA, while preserving most of the QLoRA memory benefit versus bf16. The throughput result should be reported cautiously: H8 is marginally faster than blanket QLoRA in these runs, but both low-bit policies remain about 19% slower than bf16.

Decision against the locked protocol:

- final eval loss is closer to bf16 than blanket QLoRA: supported across seeds 42-44.
- final eval loss remains inside the 1% gate versus bf16: supported.
- peak memory remains meaningfully below bf16: supported, about 25.3% lower.
- instability events do not increase: supported, zero spikes and zero NaN/Inf events.
- hardware and microbatching are matched: supported for the three 500-step RTX 3090 comparisons.

Conclusion: H8 is supported as a narrow hardware-aware selective-rescue result. It does not establish a throughput win, and the quality improvement over blanket QLoRA is modest. The scientific value is that a calibration/prediction-driven policy can be expressed through a real low-bit backend plus high-precision module rescue, improving the quality side of a memory-saving low-bit baseline while retaining nearly all of the memory benefit.

Remaining H8 work should be limited to documentation and optional robustness checks, not a new systems build-out:

- Update the project-level findings and research state with the H8 result.
- In a paper/report, present H8 as a systems feasibility extension rather than the main training contribution.
- Optional only: repeat the same selective-rescue setup on Qwen2.5-7B if direct continuity from H6.3 is needed.
- Do not fold Transformer inference policy search into H8; that should become H9 with a separate objective, cost model, and benchmark design.
