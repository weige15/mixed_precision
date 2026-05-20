#!/usr/bin/env bash
set -euo pipefail

python experiments/h1-selective-fp32-norms/code/run_lora_precision.py --precision-policy h6_custom_int8 --seed 42 --max-steps 500 --learning-rate 2e-4 --eval-every 100 --output-dir 'experiments/h6-adaptive-precision-assignment/results/train_h6_snip_style_k24_seed42_500' --fake-int8-modules $(tr '\n' ' ' < '/nfs/home/s314511048/mixed_precision/experiments/h6-adaptive-precision-assignment/results/snip_style_policy/snip_style_k24_modules.txt')
