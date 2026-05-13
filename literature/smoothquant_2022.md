# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

- **Authors:** Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
- **Year:** 2022 / ICML 2023
- **Source:** https://arxiv.org/abs/2211.10438
- **DOI:** https://doi.org/10.48550/arXiv.2211.10438

## Summary

SmoothQuant enables W8A8 INT8 LLM inference by addressing activation outliers. The central move is a mathematically equivalent transformation that migrates quantization difficulty from activations, which are hard to quantize, into weights, which are easier to quantize. This is done offline through per-channel smoothing calibrated from activation statistics.

The paper is technically a post-training quantization method, but it matters here because it gives a clean example of precision policy design from stability-relevant tensor statistics. It treats activation outliers as a first-class obstacle to low-precision LLM execution.

## Relevance To This Project

For adaptive precision assignment, SmoothQuant suggests that activation outlier scores can inform whether a module should be kept in BF16/FP32, transformed, or treated as low-risk for INT8-style execution. It also suggests that precision assignment may sometimes be paired with rescaling rather than only dtype switching.

## Key Takeaways

- LLM activations are often the hard side of INT8 quantization; weights are comparatively easier.
- Per-channel activation statistics can guide low-precision policy.
- Outlier mitigation can enable lower precision without modifying the semantic computation.

## Evidence Gaps

- SmoothQuant is inference-oriented and training-free.
- It does not evaluate LoRA fine-tuning stability, gradients, optimizer states, or loss spikes.
- Its calibration is static, so it does not answer when precision should adapt during training.
