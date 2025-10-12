from __future__ import annotations

import numpy as np

from ccpress.compression.base import BaseCompressor


def _mode_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    order = tensor.ndim
    axes = (mode,) + tuple(ax for ax in range(order) if ax != mode)
    return np.reshape(np.transpose(tensor, axes), (tensor.shape[mode], -1))


def _mode_fold(unfolded: np.ndarray, mode: int, shape: tuple[int, ...]) -> np.ndarray:
    order = len(shape)
    new_shape = list(shape)
    new_shape[mode] = unfolded.shape[0]
    axes = (mode,) + tuple(ax for ax in range(order) if ax != mode)
    tensor = np.reshape(unfolded, [unfolded.shape[0]] + [shape[ax] for ax in axes[1:]])
    inv_axes = np.argsort(axes)
    return np.transpose(tensor, inv_axes)


def _mode_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    unfolded = _mode_unfold(tensor, mode)
    result = matrix @ unfolded
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return _mode_fold(result, mode, tuple(new_shape))


class TuckerCompressor(BaseCompressor):
    """Higher-order SVD (HOSVD/Tucker) compressor."""

    name = "tucker"

    def __init__(self, ranks: tuple[int, int, int]):
        super().__init__()
        if len(ranks) != 3:
            raise ValueError("TuckerCompressor requires 3 rank components (for T, X, Y)")
        if any(r <= 0 for r in ranks):
            raise ValueError("All Tucker ranks must be positive")
        self.ranks = tuple(int(r) for r in ranks)
        self._orig_shape: tuple[int, int, int] | None = None

    def compress(self, data: np.ndarray, **kwargs):
        tensor = np.asarray(data)
        if tensor.ndim != 3:
            raise ValueError("TuckerCompressor expects a 3D array (T, X, Y)")
        self._orig_shape = tensor.shape
        ranks = tuple(min(r, dim) for r, dim in zip(self.ranks, tensor.shape))

        factors = []
        for mode, (dim, rank) in enumerate(zip(tensor.shape, ranks)):
            unfold = _mode_unfold(tensor, mode)
            U, _, _ = np.linalg.svd(unfold, full_matrices=False)
            factors.append(U[:, :rank])

        core = tensor.astype(np.float64, copy=True)
        for mode, U in enumerate(factors):
            core = _mode_product(core, U.T, mode)

        dtype = tensor.dtype
        parts = {"core": core.astype(dtype, copy=False)}
        for idx, U in enumerate(factors):
            parts[f"U{idx}"] = U.astype(dtype, copy=False)
        parts["ranks"] = np.asarray(ranks, dtype=np.int32)
        return parts

    def decompress(self, G, **kwargs):
        if not isinstance(G, dict):
            raise ValueError("TuckerCompressor expects a dict with keys 'core' and 'U{mode}'")
        core = np.asarray(G["core"])
        tensor = core.astype(np.float64, copy=True)
        factors = []
        mode = 0
        while True:
            key = f"U{mode}"
            if key not in G:
                break
            factors.append(np.asarray(G[key]))
            mode += 1
        for mode, U in enumerate(factors):
            tensor = _mode_product(tensor, U, mode)
        if self._orig_shape is not None:
            result = tensor.reshape(self._orig_shape)
        else:
            target_shape = kwargs.get("target_shape")
            result = tensor.reshape(target_shape) if target_shape is not None else tensor
        return result.astype(core.dtype, copy=False)
