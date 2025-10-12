import numpy as np
from ccpress.compression.base import BaseCompressor

class SVDCompressor(BaseCompressor):
    name = "svd"

    def __init__(self, rank: int = 50):
        super().__init__()
        self.rank = rank
        self._orig_shape = None  # 记住 (t, x, y)

    def compress(self, data: np.ndarray, **kwargs):
        t, x, y = data.shape
        self._orig_shape = (t, x, y)
        data_2d = data.reshape(t, x * y)
        U, S, Vt = np.linalg.svd(data_2d, full_matrices=False)
        return (U[:, :self.rank], S[:self.rank], Vt[:self.rank, :])

    def decompress(self, G, **kwargs):
        U_r, S_r, Vt_r = G
        approx_2d = U_r @ np.diag(S_r) @ Vt_r  # 形状 (t, x*y)
        if self._orig_shape is not None:
            t, x, y = self._orig_shape
            return approx_2d.reshape(t, x, y)
        spatial_shape = kwargs.get("spatial_shape")
        if spatial_shape:
            return approx_2d.reshape(-1, *spatial_shape)
        raise ValueError("SVDCompressor: 无法恢复形状，请传 spatial_shape 或先调用过 compress()。")
