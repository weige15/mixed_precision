# QAQ: Query-adaptive Mixed-precision Quantization for Large Language Models

- Authors: Shuxing Li, Huanrong Liu, Zelin Wang, Ruoyang Du, S. Lee, Chunlin Tian, Qingbiao Li
- Venue/year: NeurIPS 2025 ML for Systems Workshop
- Source: https://neurips.cc/virtual/2025/129098
- Topic: Query-conditioned dynamic precision for LLM inference

## Key Idea

QAQ decomposes model weights into bit planes, then uses a trainable router to
select precision based on the input query. It also supports on-demand loading
between CPU and GPU so higher-precision information is fetched only when the
query needs it.

## Relevance

This is the closest paper for the user's prompt-conditioned formulation. It
combines three pieces that matter here: input-dependent importance, bit-plane
storage, and memory movement under a hardware constraint.

## Use For This Project

- Cite for "the prompt can decide how many weight bit planes to use."
- Its bit-plane decomposition is a strong precedent for the storage solution.
- Its latency overhead is a warning: dynamic precision must be evaluated as a
  memory-latency-quality trade-off, not only as a compression ratio.

