# Evaluation Harness

## Goal

Create the CLI entrypoint that coordinates model loading, bit-plane preparation, policies, router modes, metrics, and result artifacts.

## Inputs

- `qaq/doc/proposal.md`: Results should report quality, memory, latency, and routing behavior against fixed baselines as JSONL plus summary JSON.
- `qaq/doc/detailed-design.md`: `qaq/code/evaluate.py` should expose `prepare-bitplanes`, `run-static`, `build-oracle-labels`, `train-router`, `run-router`, and `run-dynamic-loader` modes with explicit seeds and `hardware_label`.

## Tasks

- [ ] Add `qaq/code/evaluate.py` with argparse subcommands and configuration serialization into each output directory.
- [ ] Implement dataset formatting and synthetic-example support so CLI parsing and aggregation tests do not require target-model access.
- [ ] Wire model adapter, bit-plane store, policy expansion, router artifacts, and optional dynamic-loader paths behind clear subcommands.
- [ ] Write per-sample `metrics.jsonl` records with sample id, policy name, seed, hardware label, token counts, quality, latency, memory, selected bit widths, and loader mode.
- [ ] Aggregate `summary.json` with model, dataset, policies, quality, latency, memory, selected-bit distribution, dynamic-loader flag, and errors.
- [ ] Add harness tests for CLI parsing, config serialization, synthetic metrics aggregation, summary generation with loader fields, and an end-to-end toy or mocked-model run.

## Done When

- [ ] Static and router runs write repository-style `metrics.jsonl`, `summary.json`, and optional policy/router artifacts under `qaq/results/`.
- [ ] The harness can run smoke tests without target-model access and fails fast for missing target model access outside explicit smoke mode.
