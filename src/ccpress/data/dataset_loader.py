from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Tuple, Literal

def load_dat_as_memmap(dat_path: str | Path, shape: Tuple[int, int, int],
                       dtype: Literal["float32","float64"]="float32") -> np.memmap:
    """
    以内存映射的方式读取 .dat 原始三维数组 (T, X, Y)。
    不会把整块数据一次性载入内存，适合大文件。
    """
    dat_path = Path(dat_path)
    if not dat_path.exists():
        raise FileNotFoundError(dat_path)
    return np.memmap(dat_path, dtype=dtype, mode="r", shape=shape, order="C")

def raw_file_size(dat_path: str | Path) -> int:
    return Path(dat_path).stat().st_size
