from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseCompressor(ABC):
    """Abstract base class for all compressors (semantic or physical)."""

    name: str = "base"

    def __init__(self, array_uri: Optional[str] = None):
        """
        Args:
            array_uri: Optional URI for TileDB array or other storage backend.
        """
        self.array_uri = array_uri

    # ----------------------
    # 🔹 Core abstract methods
    # ----------------------
    @abstractmethod
    def compress(self, data: np.ndarray, **kwargs) -> Any:
        """
        Perform compression on input data and return compressed representation G.
        For example, G could be (U_r, S_r, V_r) for SVD.
        """
        raise NotImplementedError

    @abstractmethod
    def decompress(self, compressed_data: Any, **kwargs) -> np.ndarray:
        """
        Reconstruct (approximate) data from compressed representation.
        """
        raise NotImplementedError
    # ----------------------
    # 🔹 Utility
    # ----------------------
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, uri={self.array_uri})"
