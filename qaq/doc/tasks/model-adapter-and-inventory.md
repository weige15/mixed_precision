# Model Adapter and Inventory

## Goal

Create the shared model-inventory layer that finds target linear weights and assigns stable group ids for bit-plane storage, policies, routing, and evaluation.

## Inputs

- `qaq/doc/proposal.md`: QAQ should run as a readable local PyTorch/Transformers prototype under `qaq/`, starting with small-model smoke paths and later target-model evidence.
- `qaq/doc/detailed-design.md`: Architecture includes `qaq/code/model_adapter.py`; cross-module contracts define `tensor_name`, `module_name`, `module_role`, `layer_idx`, `group_id`, `group_granularity`, `shape`, `source_dtype`, and `target_quantized`.

## Tasks

- [ ] Add `qaq/code/model_adapter.py` with inventory records for selected Hugging Face causal-LM linear modules.
- [ ] Implement stable group-id generation for transformer-layer, attention-vs-MLP, and individual-linear granularity.
- [ ] Classify module roles and layer indices from common LLaMA/Qwen-style module names, with a safe fallback role for unknown names.
- [ ] Expose inventory serialization so runs can save the exact module list and group mapping.
- [ ] Add synthetic-module tests that verify stable ids, role assignment, shape capture, and empty-selection failure behavior.

## Done When

- [ ] Other modules can consume a deterministic inventory without loading or mutating bit-plane records.
- [ ] Tests cover all supported group granularities and reject runs with no selected target modules.
