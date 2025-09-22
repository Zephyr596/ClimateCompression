import numpy as np
import tiledb
import os
import argparse

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TILEDB_PATH = os.path.join(DATA_DIR, "arrayD")

# Supported datasets
DATASETS = {
    "4k": {
        "file": "Redsea_t2_4k_gan.dat",
        "shape": (4000, 855, 1215)
    },
    "500": {
        "file": "Redsea_t2_500_gan.dat",
        "shape": (500, 855, 1215)
    }
}

def import_to_tiledb(version: str):
    """Load raw binary data and store it in a TileDB dense array."""
    if version not in DATASETS:
        raise ValueError(f"Unknown dataset version: {version}. Choose from {list(DATASETS.keys())}")

    dataset = DATASETS[version]
    data_path = os.path.join(DATA_DIR, dataset["file"])
    shape = dataset["shape"]

    print(f"Loading {version} dataset from {data_path} ...")
    data = np.fromfile(data_path, dtype=np.float32).reshape(shape)

    # Remove existing TileDB array if it exists
    if os.path.exists(TILEDB_PATH):
        print(f"Removing existing TileDB array at {TILEDB_PATH} ...")
        tiledb.remove(TILEDB_PATH)

    # Define dimensions
    dim_t = tiledb.Dim(name="t", domain=(0, shape[0]-1), tile=100, dtype=np.int32)
    dim_x = tiledb.Dim(name="x", domain=(0, shape[1]-1), tile=64, dtype=np.int32)
    dim_y = tiledb.Dim(name="y", domain=(0, shape[2]-1), tile=64, dtype=np.int32)
    domain = tiledb.Domain(dim_t, dim_x, dim_y)

    # Define attribute
    attr = tiledb.Attr(name="temperature", dtype=np.float32, compressor=("zstd", -1))
    schema = tiledb.ArraySchema(domain=domain, attrs=[attr], sparse=False)

    # Create TileDB array
    print(f"Creating TileDB array at {TILEDB_PATH} ...")
    tiledb.DenseArray.create(TILEDB_PATH, schema)

    # Write data
    print("Writing data into TileDB ...")
    with tiledb.DenseArray(TILEDB_PATH, mode="w") as A:
        A[:] = data

    print("Import finished. TileDB array created at:", TILEDB_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Red Sea dataset into TileDB")
    parser.add_argument("--version", type=str, default="500", help="Dataset version: 500 or 4k")
    args = parser.parse_args()

    import_to_tiledb(args.version)
