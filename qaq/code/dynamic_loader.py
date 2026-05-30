"""Synchronous CPU-resident bit-plane materialization for QAQ."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from qaq.code.bitplanes import BitPlaneTensor, estimate_storage_bytes, reconstruct_tensor


@dataclass(frozen=True)
class LoaderConfig:
    target_device: str = "cpu"
    dtype: torch.dtype | None = None
    dry_run_cpu: bool = True


def validate_target_device(device: str, *, dry_run_cpu: bool = True) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        if dry_run_cpu:
            return torch.device("cpu")
        raise ValueError("CUDA dynamic loading requested but CUDA is not available")
    return target


def cpu_resident_record(record: BitPlaneTensor) -> BitPlaneTensor:
    return BitPlaneTensor(
        tensor_name=record.tensor_name,
        shape=record.shape,
        scale=record.scale,
        zero_point=record.zero_point,
        max_bits=record.max_bits,
        allowed_bit_widths=record.allowed_bit_widths,
        quantization_scheme=record.quantization_scheme,
        signed=record.signed,
        source_dtype=record.source_dtype,
        storage_dtype=record.storage_dtype,
        device_placement="cpu",
        group_id=record.group_id,
        planes=tuple(plane.detach().cpu() for plane in record.planes),
    )


def materialize_tensor(
    record: BitPlaneTensor,
    bit_width: int,
    config: LoaderConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = config or LoaderConfig()
    target = validate_target_device(cfg.target_device, dry_run_cpu=cfg.dry_run_cpu)
    cpu_record = cpu_resident_record(record)

    transfer_start = time.perf_counter()
    selected = [plane.to(target) for plane in cpu_record.planes[:bit_width]]
    transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    zero_planes = [
        torch.zeros(cpu_record.shape, dtype=cpu_record.planes[0].dtype, device=target)
        for _ in range(cpu_record.max_bits - bit_width)
    ]
    temp_record = BitPlaneTensor(
        tensor_name=cpu_record.tensor_name,
        shape=cpu_record.shape,
        scale=cpu_record.scale,
        zero_point=cpu_record.zero_point,
        max_bits=cpu_record.max_bits,
        allowed_bit_widths=cpu_record.allowed_bit_widths,
        quantization_scheme=cpu_record.quantization_scheme,
        signed=cpu_record.signed,
        source_dtype=cpu_record.source_dtype,
        storage_dtype=cpu_record.storage_dtype,
        device_placement=str(target),
        group_id=cpu_record.group_id,
        planes=tuple(selected + zero_planes),
    )

    reconstruction_start = time.perf_counter()
    tensor = reconstruct_tensor(temp_record, bit_width, dtype=cfg.dtype, device=target)
    reconstruction_ms = (time.perf_counter() - reconstruction_start) * 1000.0
    metrics = {
        "tensor_name": record.tensor_name,
        "group_id": record.group_id,
        "bit_width": bit_width,
        "device": str(target),
        "transfer_ms": transfer_ms,
        "reconstruction_ms": reconstruction_ms,
        "total_loader_ms": transfer_ms + reconstruction_ms,
        "cpu_plane_bytes": estimate_storage_bytes(record, bit_width),
        "gpu_materialized_bytes": int(tensor.numel() * tensor.element_size()),
    }
    return tensor, metrics


def materialize_policy(
    records: dict[str, BitPlaneTensor],
    group_bit_widths: dict[str, int],
    config: LoaderConfig | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    tensors: dict[str, torch.Tensor] = {}
    metrics: list[dict[str, Any]] = []
    for name, record in records.items():
        if record.group_id is None or record.group_id not in group_bit_widths:
            raise ValueError(f"No policy bit width for tensor {name} group {record.group_id}")
        tensor, row = materialize_tensor(record, int(group_bit_widths[record.group_id]), config)
        tensors[name] = tensor
        metrics.append(row)
    return tensors, metrics

