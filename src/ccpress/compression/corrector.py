import numpy as np
from typing import Literal


class ErrorCorrector:
    """
    Adaptive Error Corrector for semantic compression.

    Modes:
    - 'pointwise' : (原始方式改进版) 仅在误差超过 epsilon 时保留差值。
    - 'blockwise' : 块级均值修正，提高压缩性。
    - 'quantized' : 量化误差，降低存储成本。

    Parameters
    ----------
    epsilon : float
        允许的相对误差 (|D - D'| / vRange <= epsilon)
    mode : Literal['pointwise', 'blockwise', 'quantized']
        校正策略选择
    block_size : int
        blockwise 模式下的块大小
    dtype : np.dtype
        输出类型（float32 或 int8）
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        mode: Literal["pointwise", "blockwise", "quantized"] = "blockwise",
        block_size: int = 8,
        dtype=np.float32,
    ):
        self.epsilon = float(epsilon)
        self.mode = mode
        self.block_size = int(block_size)
        self.dtype = dtype

    # -------------------------------------------------------
    def compute(self, D: np.ndarray, D_approx: np.ndarray) -> np.ndarray:
        assert D.shape == D_approx.shape, "D and D_approx must have the same shape."

        Df = np.asarray(D, dtype=np.float64)
        Af = np.asarray(D_approx, dtype=np.float64)
        diff = Df - Af
        vRange = float(np.max(Df) - np.min(Df))
        if vRange == 0.0:
            return np.zeros_like(D, dtype=self.dtype)

        if self.mode == "pointwise":
            return self._pointwise(diff, vRange)
        elif self.mode == "blockwise":
            return self._blockwise(diff, vRange)
        elif self.mode == "quantized":
            return self._quantized(diff, vRange)
        else:
            raise ValueError(f"Unknown correction mode: {self.mode}")

    # -------------------------------------------------------
    def _pointwise(self, diff: np.ndarray, vRange: float) -> np.ndarray:
        """轻量版逐点修正"""
        mask = np.abs(diff) / vRange > self.epsilon
        E = np.zeros_like(diff)
        E[mask] = diff[mask] * 0.5  # 仅修正一半，防止E过大
        return E.astype(self.dtype, copy=False)

    # -------------------------------------------------------
    def _blockwise(self, diff: np.ndarray, vRange: float) -> np.ndarray:
        """块级均值修正"""
        E = np.zeros_like(diff)
        bs = self.block_size
        T, X, Y = diff.shape
        for t0 in range(0, T, bs):
            for x0 in range(0, X, bs):
                for y0 in range(0, Y, bs):
                    blk = diff[t0:t0+bs, x0:x0+bs, y0:y0+bs]
                    mean_err = np.mean(np.abs(blk)) / vRange
                    if mean_err > self.epsilon:
                        E[t0:t0+bs, x0:x0+bs, y0:y0+bs] = np.mean(blk)
        return E.astype(self.dtype, copy=False)

    # -------------------------------------------------------
    def _quantized(self, diff: np.ndarray, vRange: float) -> np.ndarray:
        """量化误差到 int8 范围 [-127,127]"""
        scaled = diff / (vRange * self.epsilon)
        clipped = np.clip(scaled, -1.0, 1.0)
        quantized = np.round(clipped * 127).astype(np.int8)
        return quantized

    # -------------------------------------------------------
    def __repr__(self):
        return (
            f"ErrorCorrector(epsilon={self.epsilon}, "
            f"mode='{self.mode}', block_size={self.block_size}, dtype={self.dtype})"
        )
