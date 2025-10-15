"""3D wavelet (multi-resolution) compressor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .base import BaseCompressor
from ..utils.array_codec import PackedArray, compress_int_array, decompress_int_array


def _haar_step(volume: np.ndarray) -> np.ndarray:
    """Single-level separable 3D Haar transform."""

    out = np.asarray(volume, dtype=np.float64)
    for axis in range(3):
        n = out.shape[axis]
        if n % 2 != 0:
            raise ValueError("Haar step requires even length along each axis")
        first = [slice(None)] * out.ndim
        second = [slice(None)] * out.ndim
        first[axis] = slice(0, n, 2)
        second[axis] = slice(1, n, 2)
        avg = (out[tuple(first)] + out[tuple(second)]) / 2.0
        diff = (out[tuple(first)] - out[tuple(second)]) / 2.0
        out = np.concatenate([avg, diff], axis=axis)
    return out


def _haar_inverse_step(coeff: np.ndarray) -> np.ndarray:
    recon = np.asarray(coeff, dtype=np.float64)
    for axis in reversed(range(3)):
        n = recon.shape[axis]
        half = n // 2
        first = [slice(None)] * recon.ndim
        second = [slice(None)] * recon.ndim
        first[axis] = slice(0, half)
        second[axis] = slice(half, n)
        avg = recon[tuple(first)]
        diff = recon[tuple(second)]
        shape = list(recon.shape)
        shape[axis] = n
        tmp = np.empty(shape, dtype=np.float64)
        idx0 = [slice(None)] * recon.ndim
        idx1 = [slice(None)] * recon.ndim
        idx0[axis] = slice(0, n, 2)
        idx1[axis] = slice(1, n, 2)
        tmp[tuple(idx0)] = avg + diff
        tmp[tuple(idx1)] = avg - diff
        recon = tmp
    return recon


@dataclass
class Wavelet3DCompressor(BaseCompressor):
    """Multi-level 3D Haar wavelet compressor."""

    levels: int = 2
    error_bound: float = 1e-3
    name: str = "wavelet3d"

    def __post_init__(self):
        if self.levels <= 0:
            raise ValueError("levels must be positive")
        if self.error_bound <= 0:
            raise ValueError("error_bound must be positive")

    def compress(self, data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError("Wavelet3DCompressor expects a 3D array")

        bound = float(kwargs.get("error_bound", self.error_bound))
        padded, pad_width = self._pad_to_power_of_two(data)
        coeff = padded.copy()

        active_shape = list(padded.shape)
        level_shapes: list[Tuple[int, int, int]] = []
        for _ in range(self.levels):
            level_shapes.append(tuple(active_shape))
            coeff_slices = tuple(slice(0, s) for s in active_shape)
            coeff_block = coeff[coeff_slices]
            transformed = _haar_step(coeff_block)
            coeff[coeff_slices] = transformed
            active_shape = [max(1, s // 2) for s in active_shape]
            if any(s <= 1 for s in active_shape):
                break

        levels_used = len(level_shapes)

        quantised = np.round(coeff / bound).astype(np.int32)
        packed = compress_int_array(quantised)

        return {
            "shape": np.asarray(data.shape, dtype=np.int32),
            "dtype": np.asarray([data.dtype.str.encode("ascii")], dtype="S16"),
            "pad": np.asarray(pad_width, dtype=np.int32),
            "levels": np.asarray([levels_used], dtype=np.int32),
            "level_shapes": np.asarray(level_shapes, dtype=np.int32),
            "error_bound": np.asarray([bound], dtype=np.float32),
            "coeff_shape": np.asarray(quantised.shape, dtype=np.int32),
            "coeff_dtype": np.asarray([packed.dtype.encode("ascii")], dtype="S16"),
            "coeff_len": np.asarray([packed.length], dtype=np.int64),
            "coeff": packed.data,
        }

    def decompress(self, compressed_data: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        shape = tuple(int(v) for v in compressed_data["shape"].astype(int))
        dtype_str = compressed_data["dtype"][0].decode("ascii")
        dtype = np.dtype(dtype_str)
        pad_width = tuple(int(v) for v in compressed_data["pad"].astype(int))
        levels = int(compressed_data["levels"][0])
        bound = float(compressed_data["error_bound"][0])
        coeff_shape = tuple(int(v) for v in compressed_data["coeff_shape"].astype(int))
        packed = PackedArray(
            data=np.asarray(compressed_data["coeff"], dtype=np.uint8),
            dtype=compressed_data["coeff_dtype"][0].decode("ascii"),
            length=int(compressed_data["coeff_len"][0]),
        )
        quantised = (
            decompress_int_array(packed).reshape(coeff_shape).astype(np.int32, copy=False)
        )
        level_shapes = np.asarray(compressed_data["level_shapes"], dtype=np.int32)

        coeff = quantised.astype(np.float64) * bound

        for lvl in reversed(range(levels)):
            shape_lvl = tuple(int(v) for v in level_shapes[lvl])
            coeff_slices = tuple(slice(0, s) for s in shape_lvl)
            block = coeff[coeff_slices]
            block = _haar_inverse_step(block)
            coeff[coeff_slices] = block

        if any(pad_width):
            slices = tuple(slice(0, dim) for dim in shape)
            coeff = coeff[slices]

        return coeff.astype(dtype, copy=False)

    def _pad_to_power_of_two(self, data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        pad_width = []
        for dim in data.shape:
            pow_two = 1
            while pow_two < dim:
                pow_two <<= 1
            pad_width.append(pow_two - dim)
        pad_spec = [(0, p) for p in pad_width]
        padded = np.pad(data, pad_spec, mode="edge")
        return padded, tuple(pad_width)


__all__ = ["Wavelet3DCompressor"]

