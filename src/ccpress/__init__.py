"""
ccpress: Climate Compression Framework
"""

__version__ = "0.1.0"

# Export key components for convenience
from .config import Config
from .compression import TileDBStore
from .evaluation import compression_ratio, mse_psnr_streaming

__all__ = [
    "Config",
    "TileDBStore",
    "compression_ratio",
    "mse_psnr_streaming",
]
