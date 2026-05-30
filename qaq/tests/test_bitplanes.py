from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qaq.code.bitplanes import (  # noqa: E402
    BitPlaneConfig,
    estimate_storage_bytes,
    quantize_tensor_to_bitplanes,
    reconstruct_tensor,
)


def expected_quantized(weight: torch.Tensor, max_bits: int = 8) -> torch.Tensor:
    source = weight.to(torch.float32)
    qmax = (1 << (max_bits - 1)) - 1
    max_abs = torch.max(torch.abs(source)).item()
    scale = 1.0 if max_abs == 0.0 else max_abs / qmax
    return torch.round(source / scale).clamp(-qmax, qmax) * scale


def test_signed_tensor_reconstructs_exact_quantized_value_at_max_bits() -> None:
    weight = torch.tensor([[-1.0, -0.2, 0.0, 0.3, 1.0]])
    record = quantize_tensor_to_bitplanes(weight, BitPlaneConfig(), tensor_name="signed")
    reconstructed = reconstruct_tensor(record, 8)
    assert torch.allclose(reconstructed, expected_quantized(weight), atol=1e-6)


def test_unsigned_like_tensor_reconstructs_supported_widths_and_shape() -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11.0
    record = quantize_tensor_to_bitplanes(weight, BitPlaneConfig(), tensor_name="unsigned")
    for width in (2, 4, 6, 8):
        reconstructed = reconstruct_tensor(record, width)
        assert reconstructed.shape == weight.shape


def test_zero_tensor_uses_unit_scale() -> None:
    record = quantize_tensor_to_bitplanes(torch.zeros(2, 3), BitPlaneConfig(), tensor_name="zero")
    assert record.scale == 1.0
    assert torch.equal(reconstruct_tensor(record, 8), torch.zeros(2, 3))


def test_nan_and_inf_rejected_with_tensor_name() -> None:
    with pytest.raises(ValueError, match="bad"):
        quantize_tensor_to_bitplanes(torch.tensor([float("nan")]), BitPlaneConfig(), tensor_name="bad")
    with pytest.raises(ValueError, match="bad"):
        quantize_tensor_to_bitplanes(torch.tensor([float("inf")]), BitPlaneConfig(), tensor_name="bad")


def test_non_contiguous_tensor_preserves_logical_values() -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not weight.is_contiguous()
    record = quantize_tensor_to_bitplanes(weight, BitPlaneConfig(), tensor_name="noncontig")
    assert torch.allclose(reconstruct_tensor(record, 8), expected_quantized(weight), atol=1e-6)


def test_unsupported_width_and_byte_accounting() -> None:
    record = quantize_tensor_to_bitplanes(torch.ones(3, 5), BitPlaneConfig(), tensor_name="bytes")
    assert estimate_storage_bytes(record) == 15
    assert estimate_storage_bytes(record, 4) == 8
    with pytest.raises(ValueError, match="Unsupported bit width"):
        reconstruct_tensor(record, 3)

