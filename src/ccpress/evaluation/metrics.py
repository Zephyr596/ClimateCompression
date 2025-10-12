from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Callable, Tuple

def compression_ratio(raw_bytes: int, tiledb_bytes: int) -> float:
    """raw_bytes / tiledb_bytes; 值越大越好。"""
    return float(raw_bytes) / float(tiledb_bytes) if tiledb_bytes > 0 else float("inf")

def mse_psnr_streaming(
    src: np.memmap,
    reader_fn: Callable[[Tuple[slice, slice, slice]], np.ndarray],
    shape: Tuple[int, int, int],
    block_t: int = 64,
) -> tuple[float, float]:
    """
    流式计算 MSE/PSNR：避免整块载入内存。
    reader_fn: 给出切片 (t,x,y) -> 返回对应块的 numpy 数组
    """
    T, X, Y = shape
    total = 0.0
    n = 0
    max_val = None

    for t0 in range(0, T, block_t):
        t1 = min(t0 + block_t, T)
        a = src[t0:t1, :, :]
        b = reader_fn((slice(t0, t1), slice(0, X), slice(0, Y)))
        diff = a.astype(np.float64) - b.astype(np.float64)
        total += float(np.sum(diff * diff))
        n += diff.size
        cur_max = float(np.max(np.abs(a)))
        max_val = cur_max if max_val is None else max(max_val, cur_max)

    mse = total / max(n, 1)
    # PSNR（若数据是实值场，取幅度最大值作为峰值）
    psnr = 20.0 * np.log10((max_val + 1e-12) / (np.sqrt(mse) + 1e-12)) if max_val is not None else float("nan")
    return mse, psnr
