# Hardware-Backed Selective Precision: Literature and Project Index

Date: 2026-05-23

## Question

Can module-wise precision assignment be treated as a predictor plus combinatorial optimizer that chooses per-module formats for best quality/resource trade-off, instead of relying on one global dtype or hand-picked precision islands?

## Main Takeaway

Yes. The literature strongly supports this framing, but the most mature evidence splits into three families:

1. **Mixed-precision search and constrained optimization**: HAQ, DNAS, HAWQ, HAWQ-V3, and newer LoRA/rank-bitwidth methods treat precision assignment as a search or budgeted optimization problem.
2. **Sensitivity-aware LLM quantization**: AWQ, OWQ, SpQR, SqueezeLLM, SmoothQuant, and QServe show that outliers, salient channels, sparse exceptions, and hardware-friendly formats matter more than uniform bitwidth.
3. **Backends and systems**: bitsandbytes, TorchAO, NVIDIA Transformer Engine, MS-AMP, BitBLAS/Ladder, and QServe define what policies can actually become memory or speed improvements on real hardware.

For this project, the best next formulation is not "demote a few safe bf16 modules." That saves too little. The more promising formulation is:

```text
Start from a hardware-backed low-bit baseline
then selectively rescue high-risk modules/channels/paths to bf16/fp32
under a quality and stability constraint.
```

## Most Relevant Sources

| Source | Type | Why It Matters Here | Local Note |
|---|---|---|---|
| HAQ | paper | Direct precedent for hardware-in-the-loop mixed-precision assignment with RL and resource constraints. | [haq_2019.md](haq_2019.md) |
| DNAS mixed precision | paper | Treats layer-wise bitwidth choice as differentiable architecture search over an exponential space. | [dnas_mixed_precision_quantization_2018.md](dnas_mixed_precision_quantization_2018.md) |
| AWQ | paper + repo | Activation-aware protection of salient weights; hardware-friendly INT3/INT4 LLM inference. | [awq_2023.md](awq_2023.md) |
| OWQ | paper + repo | Keeps outlier/weak columns higher precision and quantizes the rest; includes efficient fine-tuning angle. | [owq_2023.md](owq_2023.md) |
| SpQR | paper + repo | Sparse high-precision exception set plus dense 3-4 bit compression; close to selective rescue. | [spqr_2023.md](spqr_2023.md) |
| SqueezeLLM | paper + repo | Dense-and-sparse quantization with sensitivity-aware exceptions for efficient serving. | [squeezellm_2023.md](squeezellm_2023.md) |
| QServe | paper + repo | System co-design for W4A8KV4 LLM serving; important hardware-realism reference. | [qserve_2024.md](qserve_2024.md) |
| Ladder / BitBLAS | paper + repo | Mixed-precision matrix multiplication backend for quantized LLM deployment. | [ladder_bitblas_2024.md](ladder_bitblas_2024.md) |
| bitsandbytes | repo | Practical QLoRA/LLM.int8 backend already used locally; useful but not automatically faster. | [bitsandbytes_project.md](bitsandbytes_project.md) |
| TorchAO | paper + docs + repo | PyTorch-native optimization stack with PTQ/QAT/FP8/INT4/INT8/MX formats. | [torchao_project_2025.md](torchao_project_2025.md) |
| NVIDIA Transformer Engine | docs + repo | Production-grade FP8/FP4 mixed precision path for supported NVIDIA hardware. | [nvidia_transformer_engine_project.md](nvidia_transformer_engine_project.md) |
| MS-AMP | repo + docs | FP8 automatic mixed precision library tied to FP8-LM. | [ms_amp_project.md](ms_amp_project.md) |
| LoftQ | paper + repo | LoRA-fine-tuning-aware quantization; initializes LoRA to compensate quantization error. | [loftq_2023.md](loftq_2023.md) |
| QA-LoRA | paper | Quantization-aware LoRA adaptation; connects precision and adapter degrees of freedom. | [qa_lora_2023.md](qa_lora_2023.md) |
| PEQA | paper | Fine-tunes quantized models by updating quantization scales, not full weights. | [peqa_2023.md](peqa_2023.md) |
| AutoQRA | recent preprint | Very close to the user's idea: joint bitwidth and LoRA-rank optimization per layer. Treat as emerging, not established. | [autoqra_2026.md](autoqra_2026.md) |

## Implications For This Project

- The current H7 predictor should be generalized from `risk(module, fake_int8_output)` to `risk(module, format, backend)`.
- A real optimizer needs both predicted quality risk and measured hardware benefit. Per-module risk alone is not enough.
- Single-module perturbation deltas are a good starting label, but not a complete objective because module interactions are non-additive.
- The search space should be constrained before optimization: group by layer/block/role, use only backend-supported formats, and validate only a small Pareto frontier.
- The likely strongest next experiment is **selective rescue from QLoRA/NF4 or another hardware-backed baseline**, not selective demotion from bf16.

## Candidate Objective

```text
maximize      measured_memory_saving(policy) or measured_tokens_per_sec(policy)
subject to    predicted_eval_loss_delta(policy) <= 1%
              predicted_instability_risk(policy) <= threshold
              all selected formats supported by backend kernels
              policy frozen before final validation
```

## Recommended Next Literature-Aware Experiment

1. Use H6/H7 calibration and perturbation data to label high-risk modules.
2. Start with a blanket hardware-backed low-bit baseline such as QLoRA/NF4.
3. Rescue the highest-risk modules or roles to bf16/fp32 if the backend can support this without large dispatch overhead.
4. Compare against blanket QLoRA and bf16 on the same GPU, same seeds, and same 500-step protocol.

This tests whether the sensitivity predictor can improve the quality side of an already memory-saving backend.

