"""Tensor-level bit-plane quantization and reconstruction utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any

import torch


@dataclass(frozen=True)
class BitPlaneConfig:
    max_bits: int = 8
    allowed_bit_widths: tuple[int, ...] = (2, 4, 6, 8)
    quantization_scheme: str = "symmetric_per_tensor"
    storage_dtype: str = "uint8"

    def validate(self) -> None:
        if self.quantization_scheme != "symmetric_per_tensor":
            raise ValueError(f"Unsupported quantization scheme: {self.quantization_scheme}")
        if self.max_bits < 2:
            raise ValueError("max_bits must be at least 2 for signed symmetric quantization")
        invalid = [width for width in self.allowed_bit_widths if width < 2 or width > self.max_bits]
        if invalid:
            raise ValueError(f"Allowed bit widths must be between 2 and max_bits={self.max_bits}: {invalid}")


@dataclass
class BitPlaneTensor:
    tensor_name: str | None
    shape: tuple[int, ...]
    scale: float
    zero_point: int
    max_bits: int
    allowed_bit_widths: tuple[int, ...]
    quantization_scheme: str
    signed: bool
    source_dtype: str
    storage_dtype: str
    device_placement: str
    group_id: str | None
    planes: tuple[torch.Tensor, ...] = field(repr=False)

    def to_metadata_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("planes", None)
        payload["shape"] = list(self.shape)
        payload["allowed_bit_widths"] = list(self.allowed_bit_widths)
        payload["num_planes"] = len(self.planes)
        return payload


def _storage_dtype(name: str) -> torch.dtype:
    if name == "uint8":
        return torch.uint8
    if name == "bool":
        return torch.bool
    raise ValueError(f"Unsupported bit-plane storage dtype: {name}")


def _check_width(record: BitPlaneTensor, bit_width: int) -> None:
    if bit_width not in record.allowed_bit_widths:
        raise ValueError(
            f"Unsupported bit width {bit_width}; allowed widths are {list(record.allowed_bit_widths)}"
        )
    if bit_width > record.max_bits:
        raise ValueError(f"bit_width={bit_width} exceeds max_bits={record.max_bits}")


def quantize_tensor_to_bitplanes(
    weight: torch.Tensor,
    config: BitPlaneConfig,
    tensor_name: str | None = None,
    group_id: str | None = None,
) -> BitPlaneTensor:
    """Quantize a tensor and split sign+magnitude payload into MSB-first planes."""

    config.validate()
    if not torch.is_floating_point(weight):
        raise ValueError(f"Expected floating-point tensor for {tensor_name or '<unnamed>'}")
    source = weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if not torch.isfinite(source).all():
        raise ValueError(f"Tensor {tensor_name or '<unnamed>'} contains NaN or Inf values")

    max_abs = torch.max(torch.abs(source)).item() if source.numel() else 0.0
    qmax = (1 << (config.max_bits - 1)) - 1
    scale = 1.0 if max_abs == 0.0 else float(max_abs / qmax)
    quantized = torch.round(source / scale).clamp(-qmax, qmax).to(torch.int16)
    sign = (quantized < 0).to(_storage_dtype(config.storage_dtype))
    magnitude = torch.abs(quantized).to(torch.int16)

    planes: list[torch.Tensor] = [sign]
    for shift in range(config.max_bits - 2, -1, -1):
        planes.append(((magnitude >> shift) & 1).to(_storage_dtype(config.storage_dtype)))

    return BitPlaneTensor(
        tensor_name=tensor_name,
        shape=tuple(int(dim) for dim in source.shape),
        scale=scale,
        zero_point=0,
        max_bits=config.max_bits,
        allowed_bit_widths=tuple(config.allowed_bit_widths),
        quantization_scheme=config.quantization_scheme,
        signed=True,
        source_dtype=str(weight.dtype),
        storage_dtype=config.storage_dtype,
        device_placement="cpu",
        group_id=group_id,
        planes=tuple(planes),
    )


def reconstruct_tensor(
    record: BitPlaneTensor,
    bit_width: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Reconstruct a dequantized tensor from the selected MSB planes."""

    if record.quantization_scheme != "symmetric_per_tensor":
        raise ValueError(f"Unsupported quantization scheme: {record.quantization_scheme}")
    _check_width(record, bit_width)
    if len(record.planes) != record.max_bits:
        raise ValueError(f"Expected {record.max_bits} bit planes, found {len(record.planes)}")

    out_device = torch.device(device) if device is not None else record.planes[0].device
    sign = record.planes[0].to(device=out_device, dtype=torch.bool)
    magnitude = torch.zeros(record.shape, dtype=torch.int16, device=out_device)
    selected_magnitude_planes = max(bit_width - 1, 0)
    for plane_index in range(1, 1 + selected_magnitude_planes):
        shift = record.max_bits - 1 - plane_index
        magnitude = magnitude | (record.planes[plane_index].to(out_device, dtype=torch.int16) << shift)

    signed = torch.where(sign, -magnitude, magnitude).to(torch.float32)
    reconstructed = signed * float(record.scale)
    if tuple(reconstructed.shape) != tuple(record.shape):
        raise ValueError(
            f"Reconstructed shape {tuple(reconstructed.shape)} does not match metadata {record.shape}"
        )
    return reconstructed.to(dtype=dtype or torch.float32, device=out_device)


def estimate_storage_bytes(record: BitPlaneTensor, bit_width: int | None = None) -> int:
    """Estimate packed storage bytes for all planes or the selected MSB planes."""

    width = record.max_bits if bit_width is None else bit_width
    if bit_width is not None:
        _check_width(record, bit_width)
    numel = 1
    for dim in record.shape:
        numel *= int(dim)
    return int(ceil(numel * width / 8))


def quantized_dequantized_tensor(record: BitPlaneTensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return the exact max-width quantized/dequantized tensor represented by a record."""

    return reconstruct_tensor(record, record.max_bits, dtype=dtype)

