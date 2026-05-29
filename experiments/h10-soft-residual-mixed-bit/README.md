# H10 Soft Residual Mixed-Bit Static Experiment

This experiment tests the soft-pruning hypothesis directly, separate from the
current H10 GPTQ-Marlin backend baseline.

The target model is `meta-llama/Llama-3.1-8B-Instruct`. The runner focuses only
on Llama projection groups:

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`

## Representation

`code/run_static_residual_mixed_bit.py` builds one progressive residual weight
representation per targeted block:

| Assigned bits | Reconstruction |
|---:|---|
| 0 | zero reconstruction |
| 2 | first 2-bit affine residual slice |
| 4 | first slice + second 2-bit residual slice |
| 6 | first slice + second slice + third 2-bit residual slice |

This is counted as one progressive representation. It is not counted as three
separate 2-bit, 4-bit, and 6-bit checkpoints.

Blocks are column groups within each target linear projection. Calibration
forward hooks estimate per-input-channel second moments, then each block is
ranked by activation-weighted residual reconstruction error. The default soft
policy uses greedy marginal benefit per added bit for 0 -> 2 -> 4 -> 6 bit
increments. The hard policy uses the same block evidence but can only choose 0
or 4 bits.

The storage columns are reported over the targeted projection weights, not over
untouched embeddings, norms, or LM head parameters.

## Run

Use this on a CUDA host with the Instruct model available:

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/h10-soft-residual-mixed-bit/code/run_static_residual_mixed_bit.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --include-h10-gptq \
  --group-cols 128 \
  --matched-budget-bits 3.0 \
  --seq-len 512 \
  --calibration-prompts 4 \
  --eval-prompts 6 \
  --output-dir experiments/h10-soft-residual-mixed-bit/results/static_llama31_8b_instruct
```

For a local tensor-only check that does not load Hugging Face models:

```bash
python experiments/h10-soft-residual-mixed-bit/code/run_static_residual_mixed_bit.py \
  --toy-smoke \
  --group-cols 16 \
  --output-dir experiments/h10-soft-residual-mixed-bit/results/toy_smoke
```

## Current Local Status

Artifacts written:

- `results/toy_smoke/toy_smoke.json`
- `results/static_llama31_8b_instruct/run_config.json`
- `results/static_llama31_8b_instruct/summary.json`
- `results/static_llama31_8b_instruct/results_table.csv`

The local Instruct run did not complete in the default command sandbox. GPU
access is blocked there, but an escalated check sees an RTX 4050 Laptop GPU with
6141 MiB VRAM and PyTorch CUDA is available. That GPU is still too small for a
plain bf16 8B model run without CPU offload or a quantized loader, and
`meta-llama/Llama-3.1-8B-Instruct` is not present in the local Hugging Face
cache. The runner therefore recorded a machine-readable failure at model load in
`summary.json`.

The checked-in `results_table.csv` includes only the existing matched H10
bf16/GPTQ-Marlin prompt-NLL values plus pending residual rows. It should not be
used as evidence for the soft-pruning hypothesis until the command above runs
successfully on a CUDA host.

## Current Conclusion

No empirical conclusion can be drawn yet about whether soft 0/2/4/6-bit
residual pruning improves the quality-storage tradeoff over hard 0/4-bit
pruning or uniform 4-bit quantization. The residual experiment code and result
schema are in place; the exact prompt-NLL rows remain blocked by local Instruct
checkpoint availability and by the need for a CUDA/offload execution plan that
fits a 6 GB GPU.
