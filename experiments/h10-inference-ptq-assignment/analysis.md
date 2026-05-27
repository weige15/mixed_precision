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
