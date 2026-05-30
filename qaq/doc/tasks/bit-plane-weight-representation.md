# Bit-Plane Weight Representation

## Goal

Implement deterministic tensor quantization into ordered bit planes and reconstruction from selected most-significant planes.

## Inputs

- `qaq/doc/proposal.md`: Quantize selected weights to a maximum of 8 bits and reconstruct approximate weights from candidate widths such as 2, 4, 6, and 8 bits.
- `qaq/doc/detailed-design.md`: `qaq/code/bitplanes.py` owns symmetric per-tensor quantization, signedness metadata, reconstruction, storage-byte estimates, and toy-tensor tests.

## Tasks

- [ ] Add `qaq/code/bitplanes.py` with `BitPlaneTensor` metadata and a quantization config for `max_bits`, allowed widths, scheme, storage dtype, and device placement.
- [ ] Implement symmetric per-tensor quantization with zero-tensor, NaN, Inf, shape, dtype, and non-contiguous tensor handling.
- [ ] Split signed integer payloads into most-significant-first bit planes and preserve metadata needed to reconstruct negative values.
- [ ] Implement `reconstruct_tensor(record, bit_width, dtype, device)` and reject unsupported widths or schemes with clear errors.
- [ ] Implement `estimate_storage_bytes(record, bit_width=None)` for full-store and selected-width accounting.
- [ ] Add tensor-only tests for 2, 4, 6, and 8-bit reconstruction, exact max-bit quantized reconstruction, zero tensors, shape preservation, and byte accounting.

## Done When

- [ ] Tensor-level bit-plane split and reconstruction work without loading a model.
- [ ] The max-bit reconstruction matches the quantized/dequantized value and lower-bit reconstruction uses only the requested top planes.
