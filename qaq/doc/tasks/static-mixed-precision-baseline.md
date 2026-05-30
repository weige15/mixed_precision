# Static Mixed-Precision Baseline

## Goal

Provide fixed and hand-authored precision policies that expand to legal group-to-bit-width mappings for baseline comparisons.

## Inputs

- `qaq/doc/proposal.md`: Baselines must include static 8-bit, static 4-bit, simple mixed policies, no-router/random-router ablations, and oracle labels.
- `qaq/doc/detailed-design.md`: `qaq/code/policies.py` owns `PrecisionPolicy` validation, built-in policies, JSON policy loading, deterministic random baseline, and expanded policy artifacts.

## Tasks

- [ ] Add `qaq/code/policies.py` with a `PrecisionPolicy` record and an expansion function from inventory groups to bit widths.
- [ ] Implement built-in `static_8bit`, `static_4bit`, `mixed_attention_high`, `random_router_baseline`, and oracle-policy loading.
- [ ] Validate unknown group ids, missing assignments without defaults, and unsupported bit widths.
- [ ] Save expanded policy mappings and policy metadata for result artifacts.
- [ ] Add synthetic-inventory tests for fixed policies, attention/MLP mixed policies, seeded random determinism, invalid policy rejection, and JSON round trip.

## Done When

- [ ] Static and random baselines produce complete legal mappings for every selected group id.
- [ ] Policy artifacts contain fully expanded mappings so later analysis does not depend on implicit defaults.
