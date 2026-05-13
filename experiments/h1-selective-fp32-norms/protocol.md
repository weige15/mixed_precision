# H1 Protocol: fp32 Normalization During bf16 LoRA Fine-Tuning

## Hypothesis

Keeping normalization layers in fp32 during bf16 LoRA fine-tuning will reduce training instability and improve or preserve held-out validation loss relative to a standard bf16 autocast baseline, while adding less than 10% throughput overhead.

## Expected Outcome

Expected positive result: the fp32-normalization run has lower final validation negative log likelihood than the bf16 baseline, or shows fewer instability events at similar validation loss, with peak memory and tokens/sec remaining close to baseline.

Expected negative but meaningful result: standard bf16 matches fp32 normalization on validation loss and stability with better throughput, suggesting that normalization precision is not the limiting factor for small-budget LoRA fine-tuning.

## Baseline

Standard LoRA fine-tuning with PyTorch/Hugging Face bf16 mixed precision:

- Model weights loaded normally with bf16 autocast during forward pass.
- LoRA adapters trained; base model frozen.
- AdamW optimizer for trainable LoRA parameters.
- Gradient clipping at max norm 1.0.
- No special fp32 override for normalization modules beyond framework defaults.
- Fixed seed, sequence length, batch size, learning rate, optimizer steps, and dataset split.

The H1 treatment is identical except RMSNorm/LayerNorm modules execute in fp32 and return tensors cast back to the surrounding dtype.

## Mandatory Dtype-Probe Phase

Before running the main experiment, run a dtype probe under the baseline PyTorch autocast policy. This checks whether baseline autocast already runs normalization or loss-related operations in fp32.

Scientific caution: the fp32-norm treatment is only scientifically meaningful if it changes the actual compute dtype of Qwen RMSNorm / LayerNorm or another normalization operation that is not already fp32 under the baseline. If the dtype probe shows that the target normalization modules already receive fp32 inputs and produce fp32 outputs under `bf16_baseline`, then H1 should be revised before interpreting any LoRA training comparison.

Required probe command:

```bash
python experiments/h1-selective-fp32-norms/code/probe_dtypes.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --seq-len 512 \
  --dtype bf16 \
  --output-json experiments/h1-selective-fp32-norms/results/dtype_probe.json
```

## Metric

Primary metric:

- Held-out validation negative log likelihood after exactly 1,000 optimizer steps; lower is better.

Secondary metrics:

- Loss-spike count, defined as any training step where loss is greater than 2x the rolling median of the previous 50 logged steps.
- NaN/Inf count in loss or gradients.
- Peak GPU memory in GiB.
- Training throughput in tokens/sec, excluding initial model loading and first-step compilation/warmup effects.
- Final gradient norm and max observed gradient norm.

Decision rule:

- H1 is supported if fp32 normalization improves validation NLL by at least 1% relative to bf16 baseline, or removes instability events observed in bf16, while throughput degradation is under 10%.
- H1 is inconclusive if validation NLL differs by less than 1% and neither run shows instability.
- H1 is refuted for this pilot regime if fp32 normalization is no better on validation NLL or stability and costs at least 10% throughput.

## Dataset/Model

Model:

- `Qwen/Qwen2.5-0.5B`

Dataset:

- `tatsu-lab/alpaca`, using a deterministic local split:
  - train: first 8,000 examples after shuffling with seed 42
  - validation: next 1,000 examples

Fine-tuning setup:

- LoRA target modules: attention projection modules and MLP projection modules where supported by the model implementation.
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- sequence length: 512
- optimizer steps: 1,000
- learning rate: 2e-4
- batch size: largest per-device batch size that fits the GPU, with gradient accumulation used to keep effective batch size fixed at 64 sequences if possible
- precision: bf16 autocast on Ampere-or-newer CUDA GPUs

If `Qwen/Qwen2.5-0.5B` or `tatsu-lab/alpaca` cannot be loaded in the available environment, the fallback model is `EleutherAI/pythia-410m`, and the fallback dataset is `Abirate/english_quotes` formatted as causal language modeling examples. Any fallback must be recorded in the run metadata before results are interpreted.

## Exact Command To Run

Baseline:

```bash
python experiments/h1-selective-fp32-norms/code/run_lora_precision.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --precision-policy bf16_baseline \
  --seed 42 \
  --max-steps 1000 \
  --seq-len 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 64 \
  --learning-rate 2e-4 \
  --eval-every 100 \
  --output-dir experiments/h1-selective-fp32-norms/results/h1_baseline_bf16_seed42
```

Treatment:

```bash
python experiments/h1-selective-fp32-norms/code/run_lora_precision.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --dataset-name tatsu-lab/alpaca \
  --precision-policy fp32_norms \
  --seed 42 \
  --max-steps 1000 \
  --seq-len 512 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 64 \
  --learning-rate 2e-4 \
  --eval-every 100 \
  --output-dir experiments/h1-selective-fp32-norms/results/h1_fp32_norms_seed42
```

## Failure Criteria

The experiment should be marked failed, not negative, if any of the following occur:

- Baseline cannot complete 1,000 optimizer steps due to implementation errors, data loading errors, or unsupported model hooks.
- Training loss is NaN or Inf before step 50 in both baseline and treatment, suggesting a shared configuration bug.
- Validation loss cannot be computed on the same held-out split for both runs.
- Effective batch size differs between baseline and treatment without being recorded and justified.
- The precision policy changes trainable parameter count, LoRA target modules, dataset ordering, max sequence length, optimizer, learning rate, or number of optimizer steps.
- GPU memory is insufficient even after reducing per-device batch size to 1 and using gradient accumulation.
- Throughput measurement includes model download, tokenizer preprocessing, or first-step warmup time.

If the baseline completes and the treatment fails specifically because fp32 normalization causes an implementation or memory issue, record that separately as an engineering failure of the treatment, not evidence about the scientific hypothesis.
