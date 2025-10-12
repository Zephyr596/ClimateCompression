import yaml
from pathlib import Path

def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(data: dict, path: str | Path):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
