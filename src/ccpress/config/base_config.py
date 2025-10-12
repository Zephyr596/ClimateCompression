# ccpress/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict

@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    data_root: Path = None         # data 总根
    data_raw_dir: Path = None      # 原始 dat
    experiments_root: Path = None  # 所有实验的根目录（可选）

    # 数据集元信息
    datasets: Dict[str, Dict] = field(default_factory=lambda: {
        "500": {"file": "Redsea_t2_500_gan.dat", "shape": (500, 855, 1215), "dtype": "float32"},
        "4k":  {"file": "Redsea_t2_4k_gan.dat",  "shape": (4000, 855, 1215), "dtype": "float32"},
    })

    def __post_init__(self):
        self.data_root = self.project_root / "data"
        self.data_raw_dir = self.data_root / "raw"
        self.experiments_root = self.data_root  # 也可以用 self.data_root / "experiments"
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_root.mkdir(parents=True, exist_ok=True)

    # -------- 路径工厂（统一在这生成路径） --------
    def exp_root(self, exp_name: str) -> Path:
        """实验根目录：data/<exp_name>"""
        p = self.experiments_root / exp_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def make_arrayD_uri(self, exp_name: str, ds_version: str,
                        codec: str, level: int, tile: Tuple[int, int, int]) -> str:
        t, x, y = tile
        d_tag = f"redsea_t2_{ds_version}_{codec}_l{level}_tile{t}x{x}x{y}"
        uri = self.exp_root(exp_name) / "arrayD" / d_tag
        uri.parent.mkdir(parents=True, exist_ok=True)
        return str(uri)

    def make_arrayG_base(self, exp_name: str, algo: str, rank: int | None = None) -> str:
        # 例：arrayG/svd_r50 作为基底，再由 save_parts 生成 *_U/_S/_Vt
        tag = f"{algo}" + (f"_r{rank}" if rank is not None else "")
        base = self.exp_root(exp_name) / "arrayG" / tag
        base.parent.mkdir(parents=True, exist_ok=True)
        return str(base)

    def make_arrayE_uri(self, exp_name: str, epsilon: float,
                        codec: str, level: int) -> str:
        e_tag = f"eps{epsilon}".replace(".", "_")  # 避免文件名中出现多点
        uri = self.exp_root(exp_name) / "arrayE" / f"{e_tag}_{codec}_l{level}"
        uri.parent.mkdir(parents=True, exist_ok=True)
        return str(uri)

    def results_dir(self, exp_name: str) -> Path:
        p = self.exp_root(exp_name) / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p
