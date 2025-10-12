from __future__ import annotations
import numpy as np
from pathlib import Path
from ccpress.config import Config
from ccpress.utils import setup_logger, path_size_bytes
from ccpress.utils.yaml_io import load_yaml, save_yaml
from ccpress.data import load_dat_as_memmap, raw_file_size
from ccpress.compression import TileDBCompressor
from ccpress.evaluation import compression_ratio, mse_psnr_streaming

# semantic compressor (SVD etc.)
from ccpress.compression import SVDCompressor
from ccpress.compression import ErrorCorrector


def run_pipeline(cfg_file: str):
    # === Load configuration ===
    cfg_dict = load_yaml(cfg_file)
    cfg = Config()
    logger = setup_logger()

    # === Extract parameters ===
    exp = cfg_dict["experiment"]
    ds = cfg_dict["dataset"]
    td = cfg_dict["tiledb"]
    sem = cfg_dict["semantic_compression"]

    logger.info(f"Running experiment: {exp['name']}")

    # === Step 1: Load dataset ===
    meta = cfg.datasets[ds["version"]]
    dat_path = cfg.data_raw_dir / meta["file"]
    shape = tuple(meta["shape"])
    dtype = meta["dtype"]
    src = load_dat_as_memmap(dat_path, shape=shape, dtype=dtype)
    logger.info(f"Loaded dataset: {dat_path.name}, shape={shape}")

    # === Step 2: Write raw D → TileDB (arrayD) ===
    arrayD_uri = cfg.make_array_uri()
    td_comp = TileDBCompressor(
        array_uri=arrayD_uri,
        shape=shape,
        dtype=dtype,
        compressor_name=td["codec"],
        compression_level=td["level"],
        tile=tuple(td["tile"]),
        overwrite=td["overwrite"],
    )
    td_comp.write(src, block_t=td["tile"][0])
    logger.info(f"Stored arrayD → {arrayD_uri}")

    # === Step 3: Semantic compression (G) ===
    if sem["algorithm"] == "svd":
        compressor = SVDCompressor(rank=sem["rank"])
        G = compressor.compress(src)
        D_approx = compressor.decompress(G)
    else:
        raise ValueError(f"Unknown algorithm: {sem['algorithm']}")

    # === Step 4: Error correction (E) ===
    corrector = ErrorCorrector(epsilon=sem["epsilon"])
    E = corrector.compute(src, D_approx)
    D_corrected = D_approx + E
    logger.info(f"Computed correction matrix E (ε={sem['epsilon']})")

    # === Step 5: Store G/E as TileDB arrays ===
    arrayG_uri = arrayD_uri.replace("arrayD", "arrayG")
    arrayE_uri = arrayD_uri.replace("arrayD", "arrayE")

    td_G = TileDBCompressor(array_uri=arrayG_uri, shape=G.shape, dtype=str(G.dtype),
                            compressor_name=td["codec"], compression_level=td["level"],
                            tile=tuple(td["tile"]), overwrite=True)
    td_E = TileDBCompressor(array_uri=arrayE_uri, shape=E.shape, dtype=str(E.dtype),
                            compressor_name=td["codec"], compression_level=td["level"],
                            tile=tuple(td["tile"]), overwrite=True)
    td_G.write(G)
    td_E.write(E)
    logger.info(f"Stored arrayG and arrayE.")

    # === Step 6: Evaluate ===
    # raw_bytes = raw_file_size(dat_path)
    size_D = path_size_bytes(arrayD_uri)
    size_G = path_size_bytes(arrayG_uri)
    size_E = path_size_bytes(arrayE_uri)
    rho = compression_ratio(size_D, size_G + size_E)
    mse, psnr = mse_psnr_streaming(src, lambda s: D_corrected[s], shape, block_t=td["tile"][0])
    logger.info(f"Compression ratio ρ = {rho:.3f}, MSE={mse:.6e}, PSNR={psnr:.2f}")

    # === Step 7: Save results ===
    out_dir = Path(exp["output_dir"]) / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "results.yaml"
    save_yaml({
        "experiment": exp["name"],
        "rho": float(rho),
        "mse": float(mse),
        "psnr": float(psnr),
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
