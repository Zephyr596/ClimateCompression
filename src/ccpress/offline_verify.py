"""Offline verification utilities for ClimateCompression experiments.

This module provides a standalone entry-point that can be used after an
experiment has finished running.  It validates the integrity of the data
stored in TileDB (``arrayD``), re-constructs the semantic approximation using
the stored G/E components, and re-computes metrics such as compression ratio
and numerical errors against the original raw dataset.

Usage
-----

.. code-block:: bash

    python -m ccpress.offline_verify                # scan all experiments
    python -m ccpress.offline_verify -e exp_name    # verify a specific one

The verifier scans the ``data`` directory by default (``Config.experiments_root``).
You can optionally pass ``--experiment/-e`` multiple times to only validate a
subset of experiments.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tiledb

from ccpress.compression.tucker_compressor import _mode_product
from ccpress.config import Config
from ccpress.data import load_dat_as_memmap
from ccpress.evaluation import compression_ratio
from ccpress.utils import path_size_bytes, setup_logger
from ccpress.utils.yaml_io import load_yaml


@dataclass
class VerificationResult:
    experiment: str
    arrayD_mse: float
    arrayD_max_abs: float
    semantic_mse: float
    semantic_psnr: float
    semantic_max_abs: float
    compression_ratio: float


def _read_tiledb_array(uri: str) -> np.ndarray:
    """Load an entire TileDB dense array into memory."""

    with tiledb.open(uri, "r") as arr:
        data = arr[:]["v"]
    return np.asarray(data)


def _read_tiledb_block(
    uri: str,
    selection: Tuple[slice, slice, slice],
) -> np.ndarray:
    """Read a 3D block from a TileDB array."""

    with tiledb.open(uri, "r") as arr:
        return arr[selection]["v"]


def _stream_compare_with_tiledb(
    memmap: np.memmap,
    tiledb_uri: str,
    shape: Tuple[int, int, int],
    block_t: int,
) -> Tuple[float, float]:
    """Return (mse, max_abs_err) comparing memmap with TileDB array chunk by chunk."""

    total_err = 0.0
    total_n = 0
    max_abs = 0.0
    T, X, Y = shape

    with tiledb.open(tiledb_uri, "r") as arr:
        for t0 in range(0, T, block_t):
            t1 = min(t0 + block_t, T)
            sel = (slice(t0, t1), slice(0, X), slice(0, Y))
            ref = np.asarray(memmap[t0:t1]).astype(np.float64)
            blk = arr[sel]["v"].astype(np.float64)
            diff = ref - blk
            total_err += float(np.sum(diff * diff))
            total_n += diff.size
            if diff.size:
                max_abs = max(max_abs, float(np.max(np.abs(diff))))

    mse = total_err / max(total_n, 1)
    return mse, max_abs


class SemanticReconstructor:
    """Reconstruct data blocks from stored semantic components G/E."""

    def __init__(
        self,
        algo: str,
        g_paths: Dict[str, str],
        arrayE_uri: Optional[str],
        shape: Tuple[int, int, int],
        dtype: np.dtype,
    ) -> None:
        self.algo = algo.lower()
        self.arrayE_uri = arrayE_uri
        self.shape = shape
        self.dtype = dtype
        self.spatial_shape = shape[1:]

        self.G: Dict[str, np.ndarray] = {}
        for name, uri in g_paths.items():
            self.G[name] = _read_tiledb_array(uri)

        self._prepare_semantic_components()

    def _prepare_semantic_components(self) -> None:
        algo = self.algo
        if algo in {"svd", "rsvd", "randomized_svd"}:
            # Nothing to precompute; we will use blocks of U on the fly.
            self._S = self.G.get("S")
            if self._S is None:
                raise ValueError("Missing 'S' component for SVD-based experiment")
            if self.G.get("U") is None or self.G.get("Vt") is None:
                raise ValueError("Missing 'U' or 'Vt' component for SVD-based experiment")
        elif algo == "tucker":
            core = np.asarray(self.G.get("core"))
            if core.size == 0:
                raise ValueError("Tucker core tensor is empty")
            core = core.astype(np.float64)
            U1 = np.asarray(self.G.get("U1"))
            U2 = np.asarray(self.G.get("U2"))
            if U1.size == 0 or U2.size == 0:
                raise ValueError("Missing Tucker factor matrices")
            if self.G.get("U0") is None:
                raise ValueError("Missing Tucker temporal factor U0")
            # Project core along spatial modes once to speed up block computation.
            tmp = _mode_product(core, U1.astype(np.float64), 1)
            tmp = _mode_product(tmp, U2.astype(np.float64), 2)
            self._tucker_projected = tmp  # shape: (r0, X, Y)
        else:
            raise ValueError(f"Unsupported semantic algorithm: {algo}")

    def _semantic_block(self, t_slice: slice) -> np.ndarray:
        algo = self.algo
        t0, t1 = t_slice.start or 0, t_slice.stop or self.shape[0]
        n_frames = t1 - t0
        if n_frames <= 0:
            return np.zeros((0, *self.spatial_shape), dtype=self.dtype)

        if algo in {"svd", "rsvd", "randomized_svd"}:
            U = np.asarray(self.G.get("U"))[t_slice]
            S = np.asarray(self._S)
            Vt = np.asarray(self.G.get("Vt"))
            approx = (U.astype(np.float64) * S.astype(np.float64)) @ Vt.astype(np.float64)
            approx = approx.reshape(n_frames, *self.spatial_shape)
        elif algo == "tucker":
            U0 = np.asarray(self.G.get("U0"))[t_slice].astype(np.float64)
            tmp = np.tensordot(U0, self._tucker_projected, axes=([1], [0]))
            approx = tmp
        else:
            raise ValueError(f"Unsupported semantic algorithm: {algo}")

        return approx.astype(self.dtype, copy=False)

    def _error_block(self, t_slice: slice) -> Optional[np.ndarray]:
        if not self.arrayE_uri:
            return None
        X, Y = self.spatial_shape
        sel = (t_slice, slice(0, X), slice(0, Y))
        return _read_tiledb_block(self.arrayE_uri, sel)

    def read_block(self, t_slice: slice) -> np.ndarray:
        block = self._semantic_block(t_slice).astype(np.float64, copy=False)
        E = self._error_block(t_slice)
        if E is not None:
            block += E.astype(np.float64, copy=False)
        return block.astype(self.dtype, copy=False)


def _evaluate_semantic(
    src: np.memmap,
    reconstructor: SemanticReconstructor,
    shape: Tuple[int, int, int],
    block_t: int,
) -> Tuple[float, float, float]:
    """Return (mse, psnr, max_abs_err) for the semantic reconstruction."""

    total_err = 0.0
    total_n = 0
    max_abs = 0.0
    max_val = 0.0
    T, X, Y = shape

    for t0 in range(0, T, block_t):
        t1 = min(t0 + block_t, T)
        sel = slice(t0, t1)
        src_block = np.asarray(src[t0:t1]).astype(np.float64)
        recon = reconstructor.read_block(sel).astype(np.float64)
        diff = src_block - recon
        total_err += float(np.sum(diff * diff))
        total_n += diff.size
        if diff.size:
            max_abs = max(max_abs, float(np.max(np.abs(diff))))
            max_val = max(max_val, float(np.max(np.abs(src_block))))

    mse = total_err / max(total_n, 1)
    psnr = (20.0 * math.log10((max_val + 1e-12) / (math.sqrt(mse) + 1e-12))) if total_n else float("nan")
    return mse, psnr, max_abs


def verify_experiment(exp_dir: Path, cfg: Config) -> VerificationResult:
    results_file = exp_dir / "results" / "results.yaml"
    if not results_file.exists():
        raise FileNotFoundError(f"Missing results.yaml in {exp_dir}")

    result_dict = load_yaml(results_file)

    exp_name = result_dict.get("experiment") or exp_dir.name
    ds_version = result_dict.get("dataset_version") or result_dict.get("config", {}) \
        .get("dataset", {}).get("version")
    if not ds_version:
        raise ValueError(f"Cannot determine dataset version for experiment {exp_name}")

    meta = cfg.datasets[str(ds_version)]
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["dtype"])

    dat_path = cfg.data_raw_dir / meta["file"]
    if not dat_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {dat_path}")

    src = load_dat_as_memmap(dat_path, shape=shape, dtype=dtype)

    paths = result_dict.get("paths", {})
    arrayD_uri = paths.get("arrayD")
    if not arrayD_uri:
        raise ValueError(f"arrayD path missing for experiment {exp_name}")

    # Determine block size using TileDB tile size along time dimension if available.
    tile_cfg = (result_dict.get("config", {})
                .get("tiledb", {})
                .get("tile"))
    block_t = int(tile_cfg[0]) if isinstance(tile_cfg, (list, tuple)) and tile_cfg else min(64, shape[0])
    block_t = max(1, block_t)

    arrayD_mse, arrayD_max = _stream_compare_with_tiledb(src, arrayD_uri, shape, block_t)

    g_paths = paths.get("arrayG") or {}
    if not isinstance(g_paths, dict) or not g_paths:
        raise ValueError(f"arrayG paths missing or invalid for experiment {exp_name}")

    sem_cfg = (result_dict.get("config", {})
               .get("semantic_compression", {}))
    algo = str(sem_cfg.get("algorithm", "")).lower()
    if not algo:
        raise ValueError(f"Semantic algorithm not specified for {exp_name}")

    reconstructor = SemanticReconstructor(
        algo=algo,
        g_paths=g_paths,
        arrayE_uri=paths.get("arrayE"),
        shape=shape,
        dtype=dtype,
    )

    semantic_mse, semantic_psnr, semantic_max = _evaluate_semantic(src, reconstructor, shape, block_t)

    size_D = path_size_bytes(arrayD_uri)
    size_G = sum(path_size_bytes(uri) for uri in g_paths.values())
    arrayE_uri = paths.get("arrayE")
    size_E = path_size_bytes(arrayE_uri) if arrayE_uri else 0
    rho = compression_ratio(size_D, size_G + size_E)

    return VerificationResult(
        experiment=exp_name,
        arrayD_mse=arrayD_mse,
        arrayD_max_abs=arrayD_max,
        semantic_mse=semantic_mse,
        semantic_psnr=semantic_psnr,
        semantic_max_abs=semantic_max,
        compression_ratio=rho,
    )


def discover_experiments(cfg: Config, explicit: Optional[Iterable[str]] = None) -> List[Path]:
    """Return experiment directories to verify."""

    if explicit:
        dirs = []
        for name in explicit:
            path = cfg.experiments_root / name
            dirs.append(path)
        return dirs

    candidates = []
    for path in sorted(cfg.experiments_root.iterdir()):
        if path.is_dir() and (path / "results" / "results.yaml").exists():
            candidates.append(path)
    return candidates


def run_offline_verification(experiments: Optional[Iterable[str]] = None) -> List[VerificationResult]:
    cfg = Config()
    logger = setup_logger()

    exp_dirs = discover_experiments(cfg, experiments)
    if not exp_dirs:
        logger.warning("No experiments found for offline verification.")
        return []

    results: List[VerificationResult] = []
    for exp_dir in exp_dirs:
        try:
            result = verify_experiment(exp_dir, cfg)
            results.append(result)
            logger.info(
                "[%s] arrayD MSE=%.3e (max %.3e); semantic MSE=%.3e, PSNR=%.2f dB, "
                "max error=%.3e; ρ=%.3f",
                result.experiment,
                result.arrayD_mse,
                result.arrayD_max_abs,
                result.semantic_mse,
                result.semantic_psnr,
                result.semantic_max_abs,
                result.compression_ratio,
            )
        except Exception as exc:  # noqa: BLE001 - log and continue
            logger.exception("Verification failed for %s", exp_dir)
    return results


def main(argv: Optional[Sequence[str]] = None) -> List[VerificationResult]:
    parser = argparse.ArgumentParser(description="Offline verification for ClimateCompression experiments")
    parser.add_argument(
        "-e",
        "--experiment",
        action="append",
        dest="experiments",
        help="Experiment name to verify (repeatable). If omitted all experiments under data/ are checked.",
    )
    args = parser.parse_args(argv)
    return run_offline_verification(args.experiments)


if __name__ == "__main__":
    main()

