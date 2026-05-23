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

