"""ZFP-style block transform compressor.

This module introduces a lightweight ZFP-inspired compressor that operates on
fixed-size 3D blocks.  Each block undergoes a separable Haar-like transform,
its coefficients are quantised with a uniform absolute error bound, and the
integer stream is stored directly.  While the implementation omits the
bit-plane refinement and embedded coding present in the production ZFP codec,
it follows the same overall workflow, making it suitable for experimentation
within the ClimateCompression framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .base import BaseCompressor


def _haar_axis_transform(block: np.ndarray, axis: int) -> np.ndarray:
    """Apply a single-level Haar transform along the specified axis."""

    block = np.asarray(block, dtype=np.float64)
    n = block.shape[axis]
    if n % 2 != 0:
        raise ValueError("Haar transform requires even length along each axis")

    first = [slice(None)] * block.ndim
    second = [slice(None)] * block.ndim

    first[axis] = slice(0, n, 2)
    second[axis] = slice(1, n, 2)

    avg = (block[tuple(first)] + block[tuple(second)]) / 2.0
    diff = (block[tuple(first)] - block[tuple(second)]) / 2.0

    transformed = np.concatenate([avg, diff], axis=axis)
    return transformed


def _haar_axis_inverse(block: np.ndarray, axis: int) -> np.ndarray:
    n = block.shape[axis]
    half = n // 2
    first = [slice(None)] * block.ndim
    second = [slice(None)] * block.ndim

    first[axis] = slice(0, half)
    second[axis] = slice(half, n)

    avg = block[tuple(first)]
    diff = block[tuple(second)]

    # Interleave reconstruction
    shape = list(block.shape)
    shape[axis] = n
    recon = np.empty(shape, dtype=np.float64)

    idx_first = [slice(None)] * block.ndim
    idx_second = [slice(None)] * block.ndim
    idx_first[axis] = slice(0, n, 2)
    idx_second[axis] = slice(1, n, 2)

    recon[tuple(idx_first)] = avg + diff
    recon[tuple(idx_second)] = avg - diff
    return recon


def _forward_transform(block: np.ndarray) -> np.ndarray:
    transformed = np.asarray(block, dtype=np.float64)
    for axis in range(3):
        transformed = _haar_axis_transform(transformed, axis)
    return transformed


def _inverse_transform(coeff: np.ndarray) -> np.ndarray:
    recon = np.asarray(coeff, dtype=np.float64)
    for axis in reversed(range(3)):
        recon = _haar_axis_inverse(recon, axis)
    return recon


@dataclass
class ZFPTransformCompressor(BaseCompressor):
    """Simplified ZFP-style transform coder operating on 3D blocks."""

    block_size: int = 4
    error_bound: float = 1e-3
    name: str = "zfp_like"

    def __post_init__(self):
        if self.block_size <= 0 or self.block_size & (self.block_size - 1) != 0:
            raise ValueError("block_size must be a power of two")
        if self.error_bound <= 0:
            raise ValueError("error_bound must be positive")

    def compress(self, data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError("ZFPTransformCompressor expects a 3D array")

        bound = float(kwargs.get("error_bound", self.error_bound))
        if bound <= 0:
            raise ValueError("error_bound must be positive")

        padded, pad_width = self._pad_to_block(data)
        blocks = self._split_blocks(padded)

        quantised_blocks = []
        for block in blocks:
            coeff = _forward_transform(block)
            q = np.round(coeff / bound).astype(np.int32)
            quantised_blocks.append(q)

        quantised_arr = np.stack(quantised_blocks, axis=0)

        return {
            "shape": np.asarray(data.shape, dtype=np.int32),
            "dtype": np.asarray([data.dtype.str.encode("ascii")], dtype="S16"),
            "pad": np.asarray(pad_width, dtype=np.int32),
            "block_size": np.asarray([self.block_size], dtype=np.int32),
            "error_bound": np.asarray([bound], dtype=np.float32),
            "blocks": quantised_arr,
        }

    def decompress(self, compressed_data: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        shape = tuple(int(v) for v in compressed_data["shape"].astype(int))
        dtype_str = compressed_data["dtype"][0].decode("ascii")
        dtype = np.dtype(dtype_str)
        pad_width = tuple(int(v) for v in compressed_data["pad"].astype(int))
        block_size = int(compressed_data["block_size"][0])
        error_bound = float(compressed_data["error_bound"][0])
        quantised_blocks = np.asarray(compressed_data["blocks"], dtype=np.int32)

        recon_blocks = []
        for block in quantised_blocks:
            coeff = block.astype(np.float64) * error_bound
            recon_block = _inverse_transform(coeff)
            recon_blocks.append(recon_block)

        padded_shape = [
            ((dim + block_size - 1) // block_size) * block_size for dim in shape
        ]
        padded = np.zeros((len(recon_blocks), block_size, block_size, block_size), dtype=np.float64)
        for idx, block in enumerate(recon_blocks):
            padded[idx] = block

        volume = self._merge_blocks(padded, padded_shape)
        if any(pad_width):
            slices = tuple(slice(0, dim) for dim in shape)
            volume = volume[slices]

        return volume.astype(dtype, copy=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pad_to_block(self, data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        pad_width = []
        for dim in data.shape:
            remainder = dim % self.block_size
            pad_width.append(0 if remainder == 0 else self.block_size - remainder)

        pad_spec = [(0, p) for p in pad_width]
        padded = np.pad(data, pad_spec, mode="edge")
        return padded, tuple(pad_width)

    def _split_blocks(self, data: np.ndarray) -> Tuple[np.ndarray, ...]:
        b = self.block_size
        blocks = []
        for i in range(0, data.shape[0], b):
            for j in range(0, data.shape[1], b):
                for k in range(0, data.shape[2], b):
                    blocks.append(data[i : i + b, j : j + b, k : k + b])
        return tuple(blocks)

    def _merge_blocks(self, blocks: np.ndarray, padded_shape: Tuple[int, int, int]) -> np.ndarray:
        b = self.block_size
        volume = np.zeros(padded_shape, dtype=np.float64)
        idx = 0
        for i in range(0, padded_shape[0], b):
            for j in range(0, padded_shape[1], b):
                for k in range(0, padded_shape[2], b):
                    volume[i : i + b, j : j + b, k : k + b] = blocks[idx]
                    idx += 1
        return volume


__all__ = ["ZFPTransformCompressor"]

