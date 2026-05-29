# QAQ: Quality Adaptive Quantization for LLM KV Cache

- Authors: Shichen Dong, Wen Cheng, Jiayu Qin, Wei Wang
- Year: 2024; ICCV 2025 workshop version also available
- Source: https://arxiv.org/abs/2403.04643
- Code: https://github.com/ClubieDong/QAQ-KVCacheQuantization
- Topic: Quality-adaptive non-uniform KV-cache quantization

## Key Idea

QAQ quantizes the LLM KV cache non-uniformly. It treats key and value caches as
having different quantization sensitivities, adds outlier handling, and uses
attention-aware signals to protect important cache entries while compressing
less important ones.

## Why It Is Like MoBiQuant

MoBiQuant assigns weight precision from token sensitivity; QAQ assigns KV-cache
precision from cache importance and attention signals. Both are runtime
importance-aware precision allocation methods.

## Fit To Soft Pruning

QAQ is a strong cache-side analogue of soft pruning: instead of keeping or
evicting cache entries, it can preserve them with different fidelity levels.

