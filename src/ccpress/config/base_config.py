from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict

@dataclass
class Config:
    # 路径
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    data_raw_dir: Path = None
    out_dir: Path = None

    # 数据集元信息（按需添加）
    datasets: Dict[str, Dict] = field(default_factory=lambda: {
        "500": {"file": "Redsea_t2_500_gan.dat", "shape": (500, 855, 1215), "dtype": "float32"},
        "4k":  {"file": "Redsea_t2_4k_gan.dat",  "shape": (4000, 855, 1215), "dtype": "float32"},
    })
    dataset_key: str = "500"  # 默认先用500的版本

    # TileDB 参数（先用简单可跑的默认值）
    compressor_name: str = "Zstd"        # 可选：Zstd/LZ4/Zlib/Blosc 等（大小写无所谓）
    compression_level: int = 7
    tile: Tuple[int, int, int] = (32, 128, 128)  # (t, x, y) 维度的 tile 大小
    cell_order: str = "row-major"
    tile_order: str = "row-major"
    overwrite: bool = True               # 已存在时是否覆盖

    def __post_init__(self):
        self.data_raw_dir = self.project_root / "data" / "raw"
        self.out_dir = self.project_root / "data" / "arrayD"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # 生成一个易读的 Array URI 名称
    def make_array_uri(self) -> str:
        key = self.dataset_key
        t, x, y = self.tile
        name = f"redsea_t2_{key}_{self.compressor_name.lower()}_lv{self.compression_level}_tile{t}x{x}x{y}"
        return str(self.out_dir / name)
