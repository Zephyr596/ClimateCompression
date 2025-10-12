# ccpress/main.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from ccpress.config import Config
from ccpress.utils import setup_logger, path_size_bytes
from ccpress.utils.yaml_io import load_yaml, save_yaml
from ccpress.data import load_dat_as_memmap
from ccpress.compression import TileDBStore, SVDCompressor, ErrorCorrector
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
    if sem["algorithm"] == "svd":
        compressor = SVDCompressor(rank=sem["rank"])
        G = compressor.compress(src)
        D_approx = compressor.decompress(G)
        g_base_uri = cfg.make_arrayG_base(exp_name, "svd", rank=sem["rank"])
    else:
        raise ValueError(f"Unknown algorithm: {sem['algorithm']}")

    # === Step 4: Error correction (E) ===
    corrector = ErrorCorrector(epsilon=sem["epsilon"])
    E = corrector.compute(src, D_approx)
    D_corrected = D_approx + E
    logger.info(f"Computed correction matrix E (ε={sem['epsilon']})")

    # === Step 5: Store G (U/S/Vt) & E ===
    U, S, Vt = G
    tile_map = {
        "U":  None,
        "S":  None,
        "Vt": (min(256, Vt.shape[0]), min(8192, Vt.shape[1])),
    }
    g_uris = TileDBStore.save_parts(
        base_uri=g_base_uri,
        parts={"U": U, "S": S, "Vt": Vt},
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
