# Query-Conditioned Router

## Goal

Build the supervised query router path that extracts query features, creates or loads oracle labels, trains a lightweight predictor, and emits legal per-query policies.

## Inputs

- `qaq/doc/proposal.md`: Router starts with reproducible query features and can use offline oracle labels before end-to-end soft routing.
- `qaq/doc/detailed-design.md`: `qaq/code/router.py` and `qaq/code/oracle_labels.py` should support token-only features, optional model-derived features, candidate-policy scoring, policy-id classification first, artifact save/load, and router traces.

## Tasks

- [ ] Add token-only feature extraction for sample id, prompt metadata, input length, attention length, and schema version.
- [ ] Implement oracle-label selection from candidate policy scores using quality tolerance, expected cost, and a `tolerance_satisfied` flag.
- [ ] Train a lightweight policy-id classifier on features and oracle labels, with deterministic seed handling.
- [ ] Expand predicted policy ids through the policy module and validate every selected group bit width.
- [ ] Save and load router artifacts with feature-schema compatibility checks.
- [ ] Add synthetic tests for feature extraction, oracle selection including no-satisfying-candidate cases, training/prediction, policy legality, and artifact schema validation.

## Done When

- [ ] A router can be trained and evaluated on synthetic labels without model loading.
- [ ] Router predictions serialize feature records, selected policies, confidence when available, and legality-checked group bit widths.
