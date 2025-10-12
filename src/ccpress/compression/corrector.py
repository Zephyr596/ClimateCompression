import numpy as np

class ErrorCorrector:
    """Compute correction matrix E such that |D - D' + E| / vRange ≤ ε."""
    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = float(epsilon)  # 防止 YAML 给了字符串

    def compute(self, D: np.ndarray, D_approx: np.ndarray) -> np.ndarray:
        assert D.shape == D_approx.shape, "D and D_approx must have the same shape."

        Df = np.asarray(D, dtype=np.float64)
        Af = np.asarray(D_approx, dtype=np.float64)

        diff = Df - Af
        vRange = float(np.max(Df) - np.min(Df))
        if vRange == 0.0:
            # 常数场，直接返回全 0 校正
            return np.zeros_like(D, dtype=D.dtype)

        rel_err = np.abs(diff) / vRange
        mask = rel_err > self.epsilon

        E = np.zeros_like(Df)
        E[mask] = diff[mask]

        # 可选：断言（浮点放宽）
        corrected = Af + E
        new_rel_err = np.abs(Df - corrected) / vRange
        assert np.all(new_rel_err <= self.epsilon + 1e-8), "Correction failed to satisfy constraint."

        return E.astype(D.dtype, copy=False)


    def __repr__(self):
        return f"ErrorCorrector(epsilon={self.epsilon})"
