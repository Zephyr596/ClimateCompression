# ccpress/compression/tiledb_store.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple, Iterable, Dict
import numpy as np
import tiledb

_FILTER_MAP = {
    "zstd": tiledb.ZstdFilter,
    "lz4":  tiledb.LZ4Filter,
}

def _norm_tile(shape: Tuple[int, ...], tile: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if tile is None:
        return tuple(int(min(256, s)) for s in shape)
    tile = tuple(int(t) for t in tile)
    if len(tile) != len(shape):
        raise ValueError(f"tile 维度数 {len(tile)} != 数据维度数 {len(shape)}")
    return tuple(max(1, min(t, s)) for t, s in zip(tile, shape))

class TileDBStore:
    """
    通用 N 维 TileDB Dense 存取器：
    - 支持任意维度 ndarray（1D/2D/3D/...）
    - 支持按第0维分块写入（block0），便于大数据流式写
    - 提供 save_parts() 以一次性把 {"U":U, "S":S, "Vt":Vt, ...} 多个部件存为多个 TileDB 数组
    """
    name = "tiledb"

    def __init__(self,
                 array_uri: str,
                 shape: Tuple[int, ...],
                 dtype: np.dtype | str = np.float32,
                 compressor_name: str = "zstd",
                 compression_level: int = 7,
                 tile: Optional[Sequence[int]] = None,
                 cell_order: str = "row-major",
                 tile_order: str = "row-major",
                 overwrite: bool = True):
        self.array_uri = str(array_uri)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.compressor_name = compressor_name.lower()
        self.compression_level = int(compression_level)
        self.tile = _norm_tile(self.shape, tile)
        self.cell_order = cell_order
        self.tile_order = tile_order
        self.overwrite = overwrite

    # ---------- schema ----------
    def _schema(self) -> tiledb.ArraySchema:
        filt_cls = _FILTER_MAP.get(self.compressor_name)
        if not filt_cls:
            raise ValueError(f"Unsupported compressor: {self.compressor_name}")
        filters = tiledb.FilterList([filt_cls(level=self.compression_level)])

        dims = []
        for i, (n, t) in enumerate(zip(self.shape, self.tile)):
            dims.append(tiledb.Dim(name=f"d{i}", domain=(0, n-1), tile=int(t), dtype=np.uint64))
        dom = tiledb.Domain(*dims)
        attr = tiledb.Attr(name="v", dtype=self.dtype, filters=filters)
        return tiledb.ArraySchema(domain=dom, attrs=[attr], sparse=False,
                                  cell_order=self.cell_order, tile_order=self.tile_order)

    def _create_or_overwrite(self):
        uri = Path(self.array_uri)
        # 先确保父目录存在
        uri.parent.mkdir(parents=True, exist_ok=True)

        if uri.exists() and self.overwrite:
            import shutil
            shutil.rmtree(uri, ignore_errors=True)
        if not uri.exists():
            tiledb.Array.create(str(uri), self._schema())

    # ---------- write ----------
    def write(self, data: np.ndarray | np.memmap, block0: Optional[int] = None) -> None:
        """
        写入整个数组；若指定 block0，则沿第0维分块写入（每块大小为 block0）。
        """
        self._create_or_overwrite()
        data = np.asarray(data)
        if tuple(data.shape) != self.shape:
            raise ValueError(f"写入数据形状 {data.shape} 与目标 {self.shape} 不一致")

        if block0 is None or self.shape[0] <= int(block0):
            with tiledb.open(self.array_uri, "w") as A:
                A[...] = data
            return

        b = int(block0)
        n0 = self.shape[0]
        with tiledb.open(self.array_uri, "w") as A:
            for s0 in range(0, n0, b):
                e0 = min(s0 + b, n0)
                sel = (slice(s0, e0),) + tuple(slice(0, n) for n in self.shape[1:])
                A[sel] = data[sel]

    def write_blocks(self, blocks: Iterable[np.ndarray]) -> None:
        """
        按顺序写入多块（沿第0维连续），每块形状应为 (b, d1, d2, ...)，最后一块 b 可不足。
        """
        self._create_or_overwrite()
        n0 = self.shape[0]
        cursor = 0
        with tiledb.open(self.array_uri, "w") as A:
            for blk in blocks:
                blk = np.asarray(blk)
                if blk.ndim != len(self.shape):
                    raise ValueError(f"块维度 {blk.ndim} 与目标维度 {len(self.shape)} 不一致")
                if tuple(blk.shape[1:]) != self.shape[1:]:
                    raise ValueError(f"块空间形状 {blk.shape[1:]} 与目标 {self.shape[1:]} 不一致")
                b = blk.shape[0]
                if b == 0:
                    continue
                s0, e0 = cursor, cursor + b
                if e0 > n0:
                    raise ValueError(f"写入越界：到 {e0} > {n0}")
                sel = (slice(s0, e0),) + tuple(slice(0, n) for n in self.shape[1:])
                A[sel] = blk
                cursor = e0
        if cursor != n0:
            raise ValueError(f"写入数据不足：仅写到 {cursor}，目标 {n0}")

    # ---------- read ----------
    def read(self, sel: Optional[Tuple[slice, ...]] = None) -> np.ndarray:
        with tiledb.open(self.array_uri, "r") as A:
            return (A[...] if sel is None else A[sel])["v"]

    # ---------- 多部件一次性落盘（如 G = {U,S,Vt}） ----------
    @staticmethod
    def save_parts(base_uri: str,
                   parts: Dict[str, np.ndarray],
                   codec: str = "zstd",
                   level: int = 7,
                   tile_map: Optional[Dict[str, Sequence[int]]] = None,
                   overwrite: bool = True) -> Dict[str, str]:
        """
        将多个命名数组一次性存为多个 TileDB Dense 数组。
        返回 {name: uri} 映射。
        """
        base = Path(base_uri)
        base.parent.mkdir(parents=True, exist_ok=True)
        tile_map = tile_map or {}
        uris: Dict[str, str] = {}
        for name, arr in parts.items():
            arr = np.asarray(arr)
            uri = str(base.parent / f"{base.name}_{name}")
            TileDBStore(
                array_uri=uri,
                shape=tuple(arr.shape),
                dtype=arr.dtype,
                compressor_name=codec,
                compression_level=level,
                tile=tile_map.get(name),   # 可对不同部件定制 tile
                overwrite=overwrite
            ).write(arr)
            uris[name] = uri
        return uris
