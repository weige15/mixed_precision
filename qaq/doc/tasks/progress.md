# Task Progress

- [x] Model Adapter and Inventory (`qaq/doc/tasks/model-adapter-and-inventory.md`)
  - Implemented synthetic-testable linear inventory records, role classification, layer-index extraction, stable group ids, and JSON serialization.
- [x] Bit-Plane Weight Representation (`qaq/doc/tasks/bit-plane-weight-representation.md`)
  - Implemented symmetric tensor quantization, sign+magnitude MSB-first planes, reconstruction, validation, and byte estimates.
- [x] Static Mixed-Precision Baseline (`qaq/doc/tasks/static-mixed-precision-baseline.md`)
  - Implemented fixed, mixed attention-high, deterministic random, JSON load/save, and expanded artifact output.
- [x] Query-Conditioned Router (`qaq/doc/tasks/query-conditioned-router.md`)
  - Implemented token-only features, oracle label selection, nearest-centroid policy-id routing, router artifact I/O, and trace emission.
- [x] Evaluation Harness (`qaq/doc/tasks/evaluation-harness.md`)
  - Implemented argparse subcommands and synthetic smoke artifact paths for static, oracle, router, and dynamic-loader modes.
  - Target-model weight preparation/evaluation remains a final-evidence path once local ML dependencies and gated model access are available.
- [x] Dynamic Loader (`qaq/doc/tasks/dynamic-loader.md`)
  - Implemented CPU-resident records, selected-plane materialization, device validation, reconstruction through shared bit-plane API, and loader metrics.
