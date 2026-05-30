# Dynamic Loader

## Goal

Add the optional loader mode that keeps bit planes CPU-resident, materializes selected planes per query, and reports memory-latency trade-offs separately from core QAQ results.

## Inputs

- `qaq/doc/proposal.md`: CPU-to-GPU on-demand loading comes after reconstruction and routing are stable and must not claim latency wins before measurement.
- `qaq/doc/detailed-design.md`: `qaq/code/dynamic_loader.py` should synchronously transfer selected planes, reconstruct on the target device, and record loader-specific timing and memory fields.

## Tasks

- [ ] Add `qaq/code/dynamic_loader.py` with CPU-resident bit-plane records and per-query materialization from expanded policies.
- [ ] Transfer only the planes required by each selected bit width and reconstruct tensors through the shared bit-plane API.
- [ ] Measure transfer time, reconstruction time, total loader time, CPU plane bytes, and GPU materialized bytes.
- [ ] Validate target device availability and provide CPU-only dry-run behavior for tests.
- [ ] Keep loader-enabled summaries and metric fields explicitly labeled so they are not merged with non-loader latency claims.
- [ ] Add tests for CPU-only materialization, device validation, timing-field presence, and equality with non-loader reconstruction for the same bit width.

## Done When

- [ ] Dynamic-loader mode produces reconstructed tensors equivalent to non-loader reconstruction for selected bit widths.
- [ ] Loader metrics separately report memory savings and latency overhead for each enabled run.
