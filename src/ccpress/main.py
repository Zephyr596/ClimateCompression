# ccpress/main.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from ccpress.config import Config
from ccpress.utils import setup_logger, path_size_bytes
from ccpress.utils.yaml_io import load_yaml, save_yaml
from ccpress.data import load_dat_as_memmap
from ccpress.compression import (
    TileDBStore,
    SVDCompressor,
    RandomizedSVDCompressor,
    TuckerCompressor,
    ErrorCorrector,
)
from ccpress.evaluation import compression_ratio, mse_psnr_streaming

def run_pipeline(cfg_file: str):
    cfg_dict = load_yaml(cfg_file)
    cfg = Config()
    logger = setup_logger()

    exp = cfg_dict["experiment"]
    ds  = cfg_dict["dataset"]
    td  = cfg_dict["tiledb"]
    sem = cfg_dict["semantic_compression"]

    exp_name   = exp["name"]                 # e.g. "svd_eps1e-3_tile32x128x128"
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
    if algo == "svd":
        rank = int(sem.get("rank", 50))
        compressor = SVDCompressor(rank=rank)
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name,
                                          rank=rank,
                                          suffix=sem.get("suffix"))
    elif algo in {"rsvd", "randomized_svd"}:
        rank = int(sem.get("rank", 50))
        oversampling = int(sem.get("oversampling", 10))
        n_iter = int(sem.get("n_iter", 2))
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
        ranks = tuple(int(v) for v in sem.get("ranks", (20, 40, 40)))
        compressor = TuckerCompressor(ranks=ranks)
        suffix = sem.get("suffix")
        if suffix is None:
            suffix = "x".join(str(r) for r in ranks)
        g_base_uri = cfg.make_arrayG_base(exp_name, compressor.name,
                                          suffix=suffix)
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
    size_D = path_size_bytes(arrayD_uri)
    size_G = sum(path_size_bytes(uri) for uri in g_uris.values())
    size_E = path_size_bytes(arrayE_uri)
    rho = compression_ratio(size_D, size_G + size_E)
    mse, psnr = mse_psnr_streaming(src, lambda s: D_corrected[s], shape, block_t=tile[0])
    logger.info(f"Compression ratio ρ = {rho:.3f}, MSE={mse:.6e}, PSNR={psnr:.2f}")

    # === Step 7: Save results ===
    results_dir = cfg.results_dir(exp_name)
    result_file = results_dir / "results.yaml"
    save_yaml({
        "experiment": exp_name,
        "dataset_version": ds_version,
        "rho": float(rho),
        "mse": float(mse),
        "psnr": float(psnr),
        "paths": {
            "arrayD": arrayD_uri,
            "arrayG": g_uris,           # dict: {"U": "...", "S": "...", "Vt": "..."}
            "arrayE": arrayE_uri,
            "results_dir": str(results_dir),
        },
        "config": cfg_dict,
    }, result_file)
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Climate Compression Pipeline")
    parser.add_argument("--config", type=str, default="ccpress/config/experiment.yaml",
                        help="Path to YAML experiment config file.")
    args = parser.parse_args()
    run_pipeline(args.config)
