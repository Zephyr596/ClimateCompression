from pathlib import Path

def ensure_parent(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

def path_size_bytes(path: str | Path) -> int:
    """目录/文件总大小（递归）。用于评估压缩后占用。"""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    total = 0
    for sub in p.rglob("*"):
        if sub.is_file():
            total += sub.stat().st_size
    return total
