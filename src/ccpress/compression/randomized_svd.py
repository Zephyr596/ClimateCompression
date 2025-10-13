from __future__ import annotations

import numpy as np

from ccpress.compression.base import BaseCompressor


class RandomizedSVDCompressor(BaseCompressor):
    """Randomized SVD compressor following Halko et al. (2011)."""

    name = "rsvd"

    def __init__(self,
                 rank: int = 50,
                 oversampling: int = 10,
                 n_iter: int = 2,
                 random_state: int | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if oversampling < 0:
            raise ValueError("oversampling must be non-negative")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative")
        self.rank = int(rank)
        self.oversampling = int(oversampling)
        self.n_iter = int(n_iter)
        self.random_state = random_state
        self._orig_shape: tuple[int, int, int] | None = None

    def _reshape(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError("RandomizedSVDCompressor expects a 3D array (T, X, Y)")
        t, x, y = data.shape
        self._orig_shape = (t, x, y)
        return data.reshape(t, x * y)

    def compress(self, data: np.ndarray, **kwargs):
        data = np.asarray(data)
        A = self._reshape(data)
        t, n = A.shape
        r = min(self.rank, min(t, n))
        q = r + self.oversampling

        rng = np.random.default_rng(self.random_state)
        Omega = rng.standard_normal(size=(n, q))
        Y = A @ Omega
        for _ in range(self.n_iter):
            Y = A @ (A.T @ Y)
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ A
        Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ Ub[:, :r]
        dtype = data.dtype
        return {
            "U": U[:, :r].astype(dtype, copy=False),
            "S": S[:r].astype(dtype, copy=False),
            "Vt": Vt[:r, :].astype(dtype, copy=False),
        }

    def decompress(self, G, **kwargs):
        if isinstance(G, dict):
            U_r = G["U"]
            S_r = G["S"]
            Vt_r = G["Vt"]
        else:
            U_r, S_r, Vt_r = G
        approx_2d = (U_r.astype(np.float64) * S_r.astype(np.float64)) @ Vt_r.astype(np.float64)
        if self._orig_shape is not None:
            t, x, y = self._orig_shape
            return approx_2d.reshape(t, x, y).astype(U_r.dtype, copy=False)
        spatial_shape = kwargs.get("spatial_shape")
        if spatial_shape:
            return approx_2d.reshape(-1, *spatial_shape).astype(U_r.dtype, copy=False)
        raise ValueError("RandomizedSVDCompressor: cannot infer output shape.")
