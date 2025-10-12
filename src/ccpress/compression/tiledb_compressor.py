from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import tiledb
from .base import BaseCompressor

_FILTER_MAP = {
    "zstd": tiledb.ZstdFilter,
    "lz4":  tiledb.LZ4Filter,
    # "zlib": tiledb.ZlibFilter,
    # "blosc": tiledb.BloscFilter,
}

class TileDBCompressor:
    name = "tiledb"

    def __init__(self,
                 array_uri: str,
                 shape: Tuple[int, int, int],
                 dtype: np.dtype | str = np.float32,
                 compressor_name: str = "zstd",
                 compression_level: int = 7,
                 tile: Tuple[int, int, int] = (32, 128, 128),
                 cell_order: str = "row-major",
                 tile_order: str = "row-major",
                 overwrite: bool = True):
        self.array_uri = array_uri
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.compressor_name = compressor_name.lower()
        self.compression_level = int(compression_level)
        self.tile = tile
        self.cell_order = cell_order
        self.tile_order = tile_order
        self.overwrite = overwrite

    # ---------- schema ----------
    def _schema(self) -> tiledb.ArraySchema:
        t, x, y = self.shape
        tt, tx, ty = self.tile
        filt_cls = _FILTER_MAP.get(self.compressor_name)
        if not filt_cls:
            raise ValueError(f"Unsupported compressor: {self.compressor_name}")

        filters = tiledb.FilterList([filt_cls(level=self.compression_level)])
        dom = tiledb.Domain(
            tiledb.Dim(name="t", domain=(0, t-1), tile=tt, dtype=np.uint32),
            tiledb.Dim(name="x", domain=(0, x-1), tile=tx, dtype=np.uint32),
            tiledb.Dim(name="y", domain=(0, y-1), tile=ty, dtype=np.uint32),
        )
        attr = tiledb.Attr(name="v", dtype=self.dtype, filters=filters)
        return tiledb.ArraySchema(domain=dom, attrs=[attr], sparse=False,
                                  cell_order=self.cell_order, tile_order=self.tile_order)

    def _create_or_overwrite(self):
        uri = Path(self.array_uri)
        if uri.exists() and self.overwrite:
            # 小心：直接删除旧 array
            import shutil
            shutil.rmtree(uri, ignore_errors=True)
        if not uri.exists():
            tiledb.Array.create(str(uri), self._schema())

    # ---------- write & read ----------
    def write(self, data: np.ndarray | np.memmap, block_t: int | None = None) -> None:
        """
        支持分块写入，默认按 tile 的 t 维大小写入，避免一次性加载太多。
        """
        self._create_or_overwrite()
        T, X, Y = self.shape
        bt = block_t or self.tile[0]
        with tiledb.open(self.array_uri, "w") as A:
            for t0 in range(0, T, bt):
                t1 = min(t0 + bt, T)
                # 直接对 memmap 切片即可零拷贝映射
                A[t0:t1, 0:X, 0:Y] = data[t0:t1, :, :]

    def read(self) -> np.ndarray:
        """一次性读回（数据不大时方便）。大数据场景可扩展流式读取接口。"""
        with tiledb.open(self.array_uri, "r") as A:
            return A[:, :, :]["v"]
