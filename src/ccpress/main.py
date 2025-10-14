# ccpress/main.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from ccpress.config import Config
from copy import deepcopy
from typing import Any, Dict

from ccpress.utils import setup_logger, path_size_bytes
from ccpress.utils.yaml_io import load_yaml, save_yaml
from ccpress.data import load_dat_as_memmap
from ccpress.compression import (
    TileDBStore,
    SVDCompressor,
    RandomizedSVDCompressor,
    TuckerCompressor,
    SZCompressor,
    ZFPTransformCompressor,
    Wavelet3DCompressor,
    NeuralAutoencoderCompressor,
    ErrorCorrector,
)
from ccpress.evaluation import compression_ratio, mse_psnr_streaming


def _format_float(value: float) -> str:
    """Format floats for naming/logging (avoid long scientific notation)."""
    if value is None:
        return ""
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:.3g}".rstrip("0").rstrip(".")
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def generate_experiment_name(exp_cfg: Dict[str, Any], ds_cfg: Dict[str, Any],
                             td_cfg: Dict[str, Any], sem_cfg: Dict[str, Any]) -> str:
    """Automatically compose experiment names when not provided explicitly."""
    name = exp_cfg.get("name")
    if name:
        return name

    parts: list[str] = []
    version = ds_cfg.get("version")
    if version:
        parts.append(str(version))

    algo = sem_cfg.get("algorithm")
    if algo:
        parts.append(algo.lower())

    epsilon = sem_cfg.get("epsilon")
    if epsilon is not None:
        parts.append(f"eps{_format_float(float(epsilon))}")

    algo_cfg = sem_cfg.get("algorithms", {})
    if isinstance(algo_cfg, dict) and algo in algo_cfg:
        params = algo_cfg.get(algo) or {}
    else:
        params = sem_cfg

    if algo in {"svd", "rsvd", "randomized_svd"}:
        rank = params.get("rank") or sem_cfg.get("rank")
        if rank:
            parts.append(f"r{rank}")
    elif algo == "tucker":
        ranks = params.get("ranks") or sem_cfg.get("ranks")
        if ranks:
            parts.append("x".join(str(r) for r in ranks))

    tile = td_cfg.get("tile")
    if tile:
        if isinstance(tile, (list, tuple)):
            parts.append("tile" + "x".join(str(int(v)) for v in tile))
        else:
            parts.append(f"tile{tile}")

    codec = td_cfg.get("codec")
    if codec:
        parts.append(codec)

    return "_".join(parts) if parts else "experiment"

def run_experiment(cfg_dict: Dict[str, Any], *, logger=None) -> Dict[str, Any]:
    cfg = Config()
    logger = logger or setup_logger()

    cfg_dict = deepcopy(cfg_dict)

    exp = cfg_dict["experiment"]
    ds  = cfg_dict["dataset"]
    td  = cfg_dict["tiledb"]
    sem = cfg_dict["semantic_compression"]

    exp_name   = generate_experiment_name(exp, ds, td, sem)
    ds_version = ds["version"]               # "500" 或 "4k"
    codec      = td["codec"]                 # "zstd" / "lz4"
    level      = td["level"]                 # 7
    tile       = tuple(td["tile"])           # (32,128,128)

    logger.info(f"Running experiment: {exp_name}")

    # === Step 1: Load dataset ===
    meta = cfg.datasets[ds_version]
    dat_path = cfg.data_raw_dir / meta["file"]
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    src = load_dat_as_memmap(dat_path, shape=shape, dtype=dtype)
    logger.info(f"Loaded dataset: {dat_path.name}, shape={shape}")

    # === Step 2: D → TileDB ===
    arrayD_uri = cfg.make_arrayD_uri(exp_name, ds_version, codec, level, tile)
    td_store = TileDBStore(array_uri=arrayD_uri, shape=shape, dtype=dtype,
                           compressor_name=codec, compression_level=level,
                           tile=tile, overwrite=td["overwrite"])
    td_store.write(src, block0=tile[0])
    logger.info(f"Stored arrayD → {arrayD_uri}")

    # === Step 3: Semantic compression (G) ===
    algo = sem["algorithm"].lower()
    algo_cfg = sem.get("algorithms", {}) if isinstance(sem, dict) else {}
    algo_params = {}
    if isinstance(algo_cfg, dict) and algo in algo_cfg:
        algo_params = algo_cfg.get(algo) or {}

    if algo == "svd":
        rank = int(algo_params.get("rank", sem.get("rank", 50)))
        compressor = SVDCompressor(rank=rank)
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name,
                                          rank=rank,
                                          suffix=sem.get("suffix"))
    elif algo in {"rsvd", "randomized_svd"}:
        rank = int(algo_params.get("rank", sem.get("rank", 50)))
        oversampling = int(algo_params.get("oversampling", sem.get("oversampling", 10)))
        n_iter = int(algo_params.get("n_iter", sem.get("n_iter", 2)))
        compressor = RandomizedSVDCompressor(
            rank=rank,
            oversampling=oversampling,
            n_iter=n_iter,
            random_state=sem.get("random_state"),
        )
        suffix = sem.get("suffix")
        if suffix is None:
            suffix = f"os{oversampling}_it{n_iter}"
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name,
                                          rank=rank, suffix=suffix)
    elif algo == "tucker":
        ranks_cfg = algo_params.get("ranks", sem.get("ranks", (20, 40, 40)))
        ranks = tuple(int(v) for v in ranks_cfg)
        compressor = TuckerCompressor(ranks=ranks)
        suffix = sem.get("suffix")
        if suffix is None:
            suffix = "x".join(str(r) for r in ranks)
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name,
                                          suffix=suffix)
    elif algo in {"sz", "sz3", "predictq"}:
        error_bound = float(algo_params.get("error_bound", sem.get("epsilon", 1e-3)))
        mode = algo_params.get("mode", sem.get("mode", "abs"))
        compressor = SZCompressor(error_bound=error_bound, mode=mode)
        suffix = sem.get("suffix")
        if suffix is None:
            suffix = f"eps{_format_float(error_bound)}_{mode}"
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name, suffix=suffix)
    elif algo in {"zfp", "zfp_like", "zfp-transform"}:
        block = int(algo_params.get("block_size", sem.get("block_size", 4)))
        error_bound = float(algo_params.get("error_bound", sem.get("epsilon", 1e-3)))
        compressor = ZFPTransformCompressor(block_size=block, error_bound=error_bound)
        suffix = sem.get("suffix")
        if suffix is None:
            suffix = f"b{block}_eps{_format_float(error_bound)}"
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name, suffix=suffix)
    elif algo in {"wavelet", "wavelet3d"}:
        levels = int(algo_params.get("levels", sem.get("levels", 2)))
        error_bound = float(algo_params.get("error_bound", sem.get("epsilon", 1e-3)))
        compressor = Wavelet3DCompressor(levels=levels, error_bound=error_bound)
        suffix = sem.get("suffix")
        if suffix is None:
            suffix = f"L{levels}_eps{_format_float(error_bound)}"
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name, suffix=suffix)
    elif algo in {"neural", "neural_autoencoder", "nn"}:
        latent_dim = int(algo_params.get("latent_dim", sem.get("latent_dim", 64)))
        epochs = int(algo_params.get("epochs", sem.get("epochs", 200)))
        lr = float(algo_params.get("learning_rate", sem.get("learning_rate", 1e-3)))
        compressor = NeuralAutoencoderCompressor(
            latent_dim=latent_dim,
            epochs=epochs,
            learning_rate=lr,
            device=sem.get("device"),
        )
        suffix = sem.get("suffix")
        if suffix is None:
            suffix = f"ld{latent_dim}_ep{epochs}"
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name, suffix=suffix)
    else:
        raise ValueError(f"Unknown algorithm: {sem['algorithm']}")

    G = compressor.compress(src)
    D_approx = compressor.decompress(G)

    # === Step 4: Error correction (E) ===
    ec_cfg = sem.get("error_correction", {"enable": True})
    if ec_cfg.get("enable", True):
        mode = ec_cfg.get("mode", "blockwise")
        block_size = int(ec_cfg.get("block_size", 8))
        dtype_str = ec_cfg.get("dtype", "float32")
        dtype = np.dtype(dtype_str)

        corrector = ErrorCorrector(
            epsilon=sem["epsilon"],
            mode=mode,
            block_size=block_size,
            dtype=dtype,
        )
        E = corrector.compute(src, D_approx)
        D_corrected = D_approx + (
            E.astype(D_approx.dtype) if E.dtype != D_approx.dtype else E
        )
        logger.info(f"Computed correction matrix E (ε={sem['epsilon']}, mode={mode})")
    else:
        logger.info("Skipping error correction (E disabled).")
        E = np.zeros_like(src)
        D_corrected = D_approx


    # === Step 5: Store G (U/S/Vt) & E ===
    tile_map = sem.get("tile_map")

    if isinstance(tile_map, dict):
        # YAML 里可能存的是 list，需要转 tuple / None
        tile_map = {k: (tuple(v) if isinstance(v, (list, tuple)) else v)
                    for k, v in tile_map.items()}

    if isinstance(G, tuple):
        # 兼容旧格式
        parts = {name: arr for name, arr in zip(["U", "S", "Vt"], G)}
    else:
        parts = G

    if tile_map is None and algo in {"svd", "rsvd", "randomized_svd"}:
        Vt = parts.get("Vt") if isinstance(parts, dict) else None
        if Vt is not None:
            tile_map = {
                "U": None,
                "S": None,
                "Vt": (min(256, Vt.shape[0]), min(8192, Vt.shape[1])),
            }

    g_uris = TileDBStore.save_parts(
        base_uri=g_base_uri,
        parts=parts,
        codec=codec, level=level, tile_map=tile_map, overwrite=True
    )

    arrayE_uri = cfg.make_arrayE_uri(exp_name, epsilon=sem["epsilon"], codec=codec, level=level)
    td_E = TileDBStore(array_uri=arrayE_uri, shape=E.shape, dtype=E.dtype,
                       compressor_name=codec, compression_level=level,
                       tile=tile, overwrite=True)
    td_E.write(E, block0=tile[0])
    logger.info(f"Stored arrayG parts → {g_uris} and arrayE → {arrayE_uri}")

    # === Step 6: Evaluate ===
    eval_cfg = cfg_dict.get("evaluation", {})
    metrics = eval_cfg.get("metrics") or ["compression_ratio"]
    metrics = [m.lower() for m in metrics]
    metrics = list(dict.fromkeys(metrics))  # 去重并保持顺序

    metric_results: Dict[str, float] = {}
    log_parts = []

    if "compression_ratio" in metrics:
        size_D = path_size_bytes(arrayD_uri)
        size_G = sum(path_size_bytes(uri) for uri in g_uris.values())
        size_E = path_size_bytes(arrayE_uri)
        rho = compression_ratio(size_D, size_G + size_E)
        metric_results["compression_ratio"] = float(rho)
        log_parts.append(f"ρ={rho:.3f}")

    need_mse = any(m in {"mse", "psnr"} for m in metrics)
    if need_mse:
        mse, psnr = mse_psnr_streaming(src, lambda s: D_corrected[s], shape, block_t=tile[0])
        if "mse" in metrics:
            metric_results["mse"] = float(mse)
            log_parts.append(f"MSE={mse:.6e}")
        if "psnr" in metrics:
            metric_results["psnr"] = float(psnr)
            log_parts.append(f"PSNR={psnr:.2f}")

    if log_parts:
        logger.info("Metrics: " + ", ".join(log_parts))
    else:
        logger.info("No evaluation metrics requested; skipping metrics computation.")

    # === Step 7: Save results ===
    results_dir = cfg.results_dir(exp_name)
    result_file = results_dir / "results.yaml"
    save_yaml({
        "experiment": exp_name,
        "dataset_version": ds_version,
        "metrics": metric_results,
        "paths": {
            "arrayD": arrayD_uri,
            "arrayG": g_uris,           # dict: {"U": "...", "S": "...", "Vt": "..."}
            "arrayE": arrayE_uri,
            "results_dir": str(results_dir),
        },
        "config": cfg_dict,
    }, result_file)
    logger.info(f"Results saved to {result_file}")

    return {
        "experiment": exp_name,
        "metrics": metric_results,
        "paths": {
            "arrayD": arrayD_uri,
            "arrayG": g_uris,
            "arrayE": arrayE_uri,
            "results_dir": str(results_dir),
        },
    }


def run_pipeline(cfg_source: str | Dict[str, Any]):
    if isinstance(cfg_source, (str, Path)):
        cfg_dict = load_yaml(str(cfg_source))
    else:
        cfg_dict = cfg_source

    logger = setup_logger()
    return run_experiment(cfg_dict, logger=logger)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Climate Compression Pipeline")
    parser.add_argument("--config", type=str, default="ccpress/config/experiment.yaml",
                        help="Path to YAML experiment config file.")
    args = parser.parse_args()
    run_pipeline(args.config)
