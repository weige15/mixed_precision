from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qaq.code.bitplanes import BitPlaneConfig, quantize_tensor_to_bitplanes, reconstruct_tensor  # noqa: E402
from qaq.code.dynamic_loader import LoaderConfig, materialize_policy, materialize_tensor, validate_target_device  # noqa: E402


def test_cpu_materialization_equals_regular_reconstruction() -> None:
    record = quantize_tensor_to_bitplanes(
        torch.tensor([[-1.0, 0.25], [0.5, 1.0]]),
        BitPlaneConfig(),
        tensor_name="w",
        group_id="g",
    )
    tensor, metrics = materialize_tensor(record, 4, LoaderConfig(target_device="cpu"))
    assert torch.equal(tensor, reconstruct_tensor(record, 4))
    for key in ["transfer_ms", "reconstruction_ms", "total_loader_ms", "cpu_plane_bytes", "gpu_materialized_bytes"]:
        assert key in metrics


def test_cuda_validation_can_fail_without_dry_run() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            validate_target_device("cuda", dry_run_cpu=False)


def test_policy_materialization() -> None:
    record = quantize_tensor_to_bitplanes(torch.ones(2, 2), BitPlaneConfig(), tensor_name="w", group_id="g")
    tensors, metrics = materialize_policy({"w": record}, {"g": 8}, LoaderConfig(target_device="cpu"))
    assert "w" in tensors
    assert metrics[0]["bit_width"] == 8

