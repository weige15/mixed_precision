# H10 Analysis

## 2026-05-27 Re-scope to Inference PTQ

H10 is re-scoped from the earlier PEFT/QLoRA selective-rescue formulation to an
inference-side mixed-precision PTQ formulation.

Reason:

- The project plan lists mixed-precision post-training quantization, weight
  quantization, activation quantization-aware normalization, and precision
  search for different layers.
- In that plan, post-training quantization means quantizing an already trained
  model for inference deployment.
- Original HAQ is also inference-oriented, so a HAQ-inspired H10 should inherit
  the deployment setting, not only the abstract table-and-solver shape.
- The sibling pruning work is framed around sparse inference, dynamic routing,
  cache locality, kernel fusion, and hardware-friendly execution paths. The
  mixed-precision branch should complement that by deciding what precision to
  use for the computation that remains.

The earlier PEFT artifacts are not discarded. H7 calibration and perturbation
signals remain useful, and H8 selective rescue remains a side result showing
that sensitivity ranking can interact with a real backend. However, the main
H10 contribution should now connect H7-style sensitivity probes to H9-style vLLM
inference backend measurements.

Immediate next artifact:

```text
experiments/h10-inference-ptq-assignment/results/action_table.csv
```

The first table should be built from H9 results and should include unsupported
backend attempts as explicit infeasible rows. The selected policy should be
reported on a Pareto frontier rather than as a single scalar winner.

## 2026-05-27 First Inference Action Table and Solver

Implemented the first H10 inference-side action table and Pareto solver:

- `code/build_inference_action_table.py`
- `code/solve_inference_assignment.py`

The H9.1 summary artifact was regenerated from the raw benchmark and quality
directories because the previous checked-in summary had been produced from the
wrong results directory. The corrected H9.1 summary contains 21 completed
policy-workload rows and one failed policy. H9.2 still contains 18 completed
long-context rows.

The generated H10 table contains 40 rows:

- 39 completed backend-real policy-workload rows from H9.1 and H9.2.
- 1 infeasible backend row for `fp16_torchao`, retained with its failure reason.

The solver enforces:

- backend feasible,
- prompt-NLL quality risk `<= 0.01`,
- at least one deployment metric improving versus `bf16_default`,
- Pareto non-domination within each workload.

Current solver output:

| Count | Value |
|---|---:|
| Input action rows | 40 |
| Feasible rows passing quality gate | 36 |
| Accepted rows before Pareto filtering | 13 |
| Selected Pareto rows | 8 |

Interpretation:

- `fp16_default` remains the cleanest low-risk candidate: it appears on the
  selected frontier for H9.1 prefill/mixed workloads and H9.2 prefill/batch
  workloads, with zero measured memory cost.
- FP8 KV-cache policies can appear on the workload frontier when they buy
  latency, especially long-context prefill/batch cases, but they still carry
  about `+0.8 GiB` measured memory cost on this RTX 3090/vLLM stack.
- `fp16_bitsandbytes` is rejected by the 1% quality gate despite its decode
  speedup.
- `fp16_torchao` is not considered selectable because vLLM requires an explicit
  `torchao_config`.

The active next research step is not another global KV-cache toggle. The code
path is now ready to ingest richer backend-real PTQ candidates such as concrete
TorchAO configs or AWQ/GPTQ/Marlin artifacts, then re-run the same table and
solver pipeline.

## 2026-05-27 TorchAO H9 Candidate Hook

The first richer backend-real PTQ candidates have been added upstream in H9:

- `fp16_torchao_int8wo`
- `fp16_torchao_int8dyn_int8w`
- `fp16_torchao_int4wo_g128`

They are not H10 evidence yet because this shell cannot initialize CUDA/NVML and
no benchmark or quality artifacts exist for them. Once a CUDA-host smoke run
and full H9 benchmark/quality passes complete, the existing H10 table builder
will ingest those new artifacts through the regenerated H9 summaries.
