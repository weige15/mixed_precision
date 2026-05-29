# FineQ: Software-Hardware Co-Design for Low-Bit Fine-Grained Mixed-Precision Quantization of LLMs

- Authors: Xilong Xie, Liang Wang, Limin Xiao, Meng Han, Lin Sun, Shuai Zheng, Xiangrong Xu
- Year: 2025
- Source: https://arxiv.org/abs/2504.19746
- Topic: Fine-grained mixed precision with aligned memory access and accelerator support

## Key Idea

FineQ partitions weights into finer clusters, protects outliers within clusters,
and designs an encoding scheme that concatenates index and data to support
aligned memory access. It pairs the algorithm with accelerator support for the
resulting mixed-precision layout.

## Relevance

This is a strong storage-and-hardware paper. It is not enough to decide that a
few values need more bits; the representation also needs indices, packing, and
aligned access so the system does not lose the benefit to metadata and
irregular memory movement.

## Use For This Project

- Cite for the claim that heterogeneous bitwidth storage is a systems problem.
- Use its outlier-protection-plus-cluster framing as a hardware-friendly version
  of "important information gets more bits."
- Helps define evaluation metrics beyond quality: metadata overhead, alignment,
  load efficiency, and accelerator compatibility.

