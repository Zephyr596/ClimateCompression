import numpy as np
from ccpress.compression.base import BaseCompressor

class SVDCompressor(BaseCompressor):
    name = "svd"

    def __init__(self, rank: int = 50):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = int(rank)
        self._orig_shape: tuple[int, int, int] | None = None  # (t, x, y)

    def compress(self, data: np.ndarray, **kwargs):
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError("SVDCompressor expects a 3D array (T, X, Y)")
        t, x, y = data.shape
        self._orig_shape = (t, x, y)
        data_2d = data.reshape(t, x * y)
        U, S, Vt = np.linalg.svd(data_2d, full_matrices=False)
        r = min(self.rank, U.shape[1])
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
        raise ValueError("SVDCompressor: 无法恢复形状，请传 spatial_shape 或先调用过 compress()。")
