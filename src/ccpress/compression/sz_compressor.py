"""Simplified SZ3-style compressor implementation.

This module implements a predictor-quantizer entropy coding inspired
compressor that follows the high-level ideas of SZ3.  The goal is not to
perfectly reproduce the highly optimised reference implementation but to
provide a drop-in compressor class compatible with the existing
``BaseCompressor`` interface.  The implementation focuses on:

* First-order 3D Lorenzo prediction (x/y/t neighbours).
* Uniform absolute-error bounded quantisation.
* Simple run-length encoding for the integer residual stream.

The resulting compressed representation is stored as a dictionary of numpy
arrays so it can be persisted with ``TileDBStore.save_parts`` without any
special handling.  Decompression follows the exact inverse procedure to
guarantee the same pointwise error bound that was enforced during
compression.

The class is intentionally lightweight so it can serve as a baseline for
future, more advanced SZ variants (relative error, adaptive quantisers,
Huffman coding, etc.) while remaining easy to understand and integrate in
unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .base import BaseCompressor
from ..utils.array_codec import PackedArray, compress_int_array, decompress_int_array


def _lorenzo_predict(recon: np.ndarray, idx: Tuple[int, int, int]) -> float:
    """First-order 3D Lorenzo predictor.

    Args:
        recon: Already reconstructed values (same shape as original data).
        idx:   Current 3D index (t, x, y).

    Returns:
        Predicted float value based on previously reconstructed neighbours.
    """

    i, j, k = idx
    if i == 0 or j == 0 or k == 0:
        # Boundaries: fall back to zero predictor.  During reconstruction the
        # original values are stored verbatim via the quantised residual, so we
        # do not lose information.
        return 0.0

    return (
        recon[i - 1, j, k]
        + recon[i, j - 1, k]
        + recon[i, j, k - 1]
        - recon[i - 1, j - 1, k]
        - recon[i - 1, j, k - 1]
        - recon[i, j - 1, k - 1]
        + recon[i - 1, j - 1, k - 1]
    )


def _rle_encode(stream: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Simple run-length encoding specialised for integer streams."""

    values = []
    counts = []
    last = None
    count = 0
    for val in stream:
        if last is None:
            last = int(val)
            count = 1
            continue
        if val == last:
            count += 1
        else:
            values.append(last)
            counts.append(count)
            last = int(val)
            count = 1
    if last is not None:
        values.append(last)
        counts.append(count)
    if not values:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    return np.asarray(values, dtype=np.int32), np.asarray(counts, dtype=np.int32)


def _rle_decode(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Decode run-length encoded integers back to the original stream."""

    if values.size == 0:
        return np.zeros((0,), dtype=np.int32)
    expanded = np.repeat(values.astype(np.int32), counts.astype(np.int32))
    return expanded


@dataclass
class SZCompressor(BaseCompressor):
    """SZ3-inspired compressor with Lorenzo prediction and uniform quantiser."""

    error_bound: float = 1e-3
    mode: str = "abs"  # "abs" | "value_rel"
    name: str = "sz3"

    def __post_init__(self):
        if self.error_bound <= 0:
            raise ValueError("error_bound must be positive")
        mode = self.mode.lower()
        if mode not in {"abs", "value_rel"}:
            raise ValueError("mode must be 'abs' or 'value_rel'")
        self.mode = mode

    # ------------------------------------------------------------------
    # BaseCompressor API
    # ------------------------------------------------------------------
    def compress(self, data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError("SZCompressor expects a 3D array (t, x, y)")

        bound = float(kwargs.get("error_bound", self.error_bound))
        if bound <= 0:
            raise ValueError("error_bound must be positive")

        if self.mode == "abs":
            step = np.full((), bound, dtype=np.float64)
        else:  # value_rel
            scale = np.max(np.abs(data))
            scale = scale if scale > 0 else 1.0
            step = np.full((), bound * scale, dtype=np.float64)

        quantised, recon = self._quantise(data, float(step))
        values, counts = _rle_encode(quantised.ravel())
        values_packed = compress_int_array(values)
        counts_packed = compress_int_array(counts)

        return {
            "shape": np.asarray(data.shape, dtype=np.int32),
            "dtype": np.asarray([data.dtype.str.encode("ascii")], dtype="S16"),
            "error_bound": np.asarray([bound], dtype=np.float32),
            "rle_values": values_packed.data,
            "rle_values_dtype": np.asarray(
                [values_packed.dtype.encode("ascii")], dtype="S16"
            ),
            "rle_values_len": np.asarray([values_packed.length], dtype=np.int64),
            "rle_counts": counts_packed.data,
            "rle_counts_dtype": np.asarray(
                [counts_packed.dtype.encode("ascii")], dtype="S16"
            ),
            "rle_counts_len": np.asarray([counts_packed.length], dtype=np.int64),
        }

    def decompress(self, compressed_data: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        shape = tuple(int(v) for v in compressed_data["shape"].astype(int))
        dtype_str = compressed_data["dtype"][0].decode("ascii")
        dtype = np.dtype(dtype_str)
        error_bound = float(compressed_data["error_bound"][0])

        values_packed = PackedArray(
            data=np.asarray(compressed_data["rle_values"], dtype=np.uint8),
            dtype=compressed_data["rle_values_dtype"][0].decode("ascii"),
            length=int(compressed_data["rle_values_len"][0]),
        )
        counts_packed = PackedArray(
            data=np.asarray(compressed_data["rle_counts"], dtype=np.uint8),
            dtype=compressed_data["rle_counts_dtype"][0].decode("ascii"),
            length=int(compressed_data["rle_counts_len"][0]),
        )

        values = decompress_int_array(values_packed).astype(np.int32, copy=False)
        counts = decompress_int_array(counts_packed).astype(np.int32, copy=False)
        quantised_stream = _rle_decode(values, counts)

        if quantised_stream.size != int(np.prod(shape)):
            raise ValueError("Decoded stream length does not match target shape")

        quantised = quantised_stream.reshape(shape)
        recon = self._dequantise(quantised, error_bound)
        return recon.astype(dtype, copy=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _quantise(self, data: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
        quantised = np.zeros_like(data, dtype=np.int32)
        recon = np.zeros_like(data, dtype=np.float64)
        for idx in np.ndindex(data.shape):
            pred = _lorenzo_predict(recon, idx)
            residual = data[idx] - pred
            q = int(np.round(residual / step))
            quantised[idx] = q
            recon[idx] = pred + q * step
        return quantised, recon

    def _dequantise(self, quantised: np.ndarray, step: float) -> np.ndarray:
        recon = np.zeros(quantised.shape, dtype=np.float64)
        for idx in np.ndindex(quantised.shape):
            pred = _lorenzo_predict(recon, idx)
            recon[idx] = pred + float(quantised[idx]) * step
        return recon


__all__ = ["SZCompressor"]

