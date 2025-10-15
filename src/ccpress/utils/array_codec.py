"""Utility helpers for compact integer array serialisation.

The climate compression baselines often quantise floating-point data to
integers which are then stored inside dictionaries handled by the
``TileDBStore``.  Keeping these arrays in their raw ``int32`` form results in
very poor compression ratios.  This module provides lightweight helper
functions to pack those integer arrays into bytes using ``zlib`` while also
selecting the smallest integer dtype that can faithfully represent the data.

The helpers are intentionally tiny and dependency free so they can be reused
by multiple compressor implementations without pulling in any third-party
codecs.  They simply convert the numpy arrays to the chosen dtype, compress
the raw bytes with ``zlib``, and expose metadata so the original array can be
reconstructed losslessly during decompression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zlib


@dataclass(frozen=True)
class PackedArray:
    """Container describing a compressed integer numpy array."""

    data: np.ndarray
    dtype: str
    length: int


def _select_minimal_int_dtype(min_val: int, max_val: int) -> np.dtype:
    """Return the narrowest integer dtype able to encode the value range."""

    if min_val >= 0:
        if max_val <= np.iinfo(np.uint8).max:
            return np.dtype(np.uint8)
        if max_val <= np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        if max_val <= np.iinfo(np.uint32).max:
            return np.dtype(np.uint32)
        return np.dtype(np.uint64)

    if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
        return np.dtype(np.int8)
    if min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def compress_int_array(array: np.ndarray) -> PackedArray:
    """Compress an integer numpy array into a packed byte representation."""

    arr = np.asarray(array)
    length = int(arr.size)

    if length == 0:
        dtype = np.dtype(np.int32)
        return PackedArray(np.zeros((0,), dtype=np.uint8), dtype.str, 0)

    min_val = int(arr.min())
    max_val = int(arr.max())
    dtype = _select_minimal_int_dtype(min_val, max_val)
    cast = arr.astype(dtype, copy=False)
    payload = zlib.compress(cast.tobytes(), level=3)
    packed = np.frombuffer(payload, dtype=np.uint8)
    return PackedArray(packed, dtype.str, length)


def decompress_int_array(packed: PackedArray) -> np.ndarray:
    """Inverse of :func:`compress_int_array`."""

    if packed.length == 0:
        return np.zeros((0,), dtype=np.int32)

    payload = bytes(np.asarray(packed.data, dtype=np.uint8))
    raw = zlib.decompress(payload)
    array = np.frombuffer(raw, dtype=np.dtype(packed.dtype), count=packed.length)
    return array.copy()


__all__ = ["PackedArray", "compress_int_array", "decompress_int_array"]

