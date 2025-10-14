"""Utilities for sweeping ClimateCompression experiments.

This module centralises the logic required by both the Streamlit GUI and a
command line interface so that adding new compression algorithms only needs to
be done in a single location.  The public entry points are:

``run_cli``
    Execute a sweep from the command line.

``sweep_runs``
    Generator that yields results for each combination of algorithm and epsilon
    so callers can render progress information while experiments are running.

The module keeps track of the canonical algorithm names supported by
``run_experiment`` and transparently maps aliases to the canonical form.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import traceback

import pandas as pd

from .config import Config
from .main import run_experiment
from .utils.yaml_io import load_yaml

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

# Canonical sweepable algorithms and their associated metadata.  ``value`` is
# the string that should be written into the experiment configuration.  The
# optional ``aliases`` entry lists alternative names that map to the canonical
# key.
SWEEP_ALGORITHMS: Dict[str, Dict[str, object]] = {
    "svd": {
        "label": "Truncated SVD",
        "value": "svd",
        "aliases": {"svd"},
    },
    "rsvd": {
        "label": "Randomised SVD",
        "value": "rsvd",
        "aliases": {"rsvd", "randomized_svd"},
    },
    "tucker": {
        "label": "Tucker Decomposition",
        "value": "tucker",
        "aliases": {"tucker"},
    },
    "sz": {
        "label": "SZ3 Predictor-Quantiser",
        "value": "sz",
        "aliases": {"sz", "sz3", "predictq"},
    },
    "zfp": {
        "label": "ZFP Transform",
        "value": "zfp",
        "aliases": {"zfp", "zfp_like", "zfp-transform"},
    },
    "wavelet3d": {
        "label": "3D Wavelet",
        "value": "wavelet3d",
        "aliases": {"wavelet", "wavelet3d"},
    },
    "neural": {
        "label": "Neural Autoencoder",
        "value": "neural_autoencoder",
        "aliases": {"neural", "neural_autoencoder", "nn"},
    },
}

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for key, meta in SWEEP_ALGORITHMS.items():
    aliases = set(meta.get("aliases", set())) | {key}
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = key


# ---------------------------------------------------------------------------
# Helper functions shared across GUI and CLI
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    return load_yaml(str(cfg_path))


def normalise_algorithm(name: str | None) -> str | None:
    """Return the canonical algorithm key when known, otherwise lower-case."""

    if not name:
        return None
    key = name.strip().lower()
    return _ALIAS_TO_CANONICAL.get(key, key)


def algorithm_label(name: str) -> str:
    """Human readable label for display purposes."""

    meta = SWEEP_ALGORITHMS.get(name)
    if not meta:
        return name
    label = str(meta.get("label", name))
    return f"{label} ({name})"


def algorithm_value(name: str) -> str:
    """Return the value that should be written into the configuration."""

    meta = SWEEP_ALGORITHMS.get(name)
    if meta:
        return str(meta.get("value", name))
    return name


def default_sweep_values(cfg_dict: Dict) -> tuple[list[str], list[float]]:
    """Infer default algorithm/epsilon sweeps from the configuration."""

    sem = cfg_dict.get("semantic_compression", {}) or {}
    sweep_cfg = cfg_dict.get("sweep", {}) or {}

    available_algorithms: list[str] = []
    algo_cfg = sem.get("algorithms", {})
    if isinstance(algo_cfg, dict):
        for name in algo_cfg.keys():
            norm = normalise_algorithm(name)
            if norm and norm not in available_algorithms:
                available_algorithms.append(norm)

    default_algo = normalise_algorithm(sem.get("algorithm"))
    if default_algo and default_algo not in available_algorithms:
        available_algorithms.append(default_algo)

    sweep_algos_cfg = sweep_cfg.get("algorithms")
    if sweep_algos_cfg:
        sweep_algorithms = [
            algo
            for algo in (normalise_algorithm(a) for a in sweep_algos_cfg)
            if algo
        ]
    else:
        sweep_algorithms = available_algorithms or (
            [default_algo] if default_algo else []
        )

    eps_cfg = sweep_cfg.get("epsilons")
    if eps_cfg:
        epsilon_values = [float(eps) for eps in eps_cfg]
    else:
        epsilon = sem.get("epsilon", 1e-3)
        epsilon_values = [float(epsilon)]

    unique_algorithms = list(dict.fromkeys(sweep_algorithms))
    return unique_algorithms, epsilon_values


def all_known_algorithms(additional: Iterable[str] | None = None) -> list[str]:
    """Return a deduplicated list of algorithms including known ones."""

    ordered: list[str] = []
    for source in (additional or []):
        if source and source not in ordered:
            ordered.append(source)
    for key in SWEEP_ALGORITHMS.keys():
        if key not in ordered:
            ordered.append(key)
    return ordered


def resolve_metrics(cfg_dict: Dict, metrics: Optional[Sequence[str]] = None) -> list[str]:
    eval_cfg = cfg_dict.get("evaluation", {}) or {}
    if metrics:
        resolved = [m.lower() for m in metrics]
    else:
        resolved = [m.lower() for m in eval_cfg.get("metrics", ["compression_ratio"])]
    return list(dict.fromkeys(resolved))


def sweep_runs(
    cfg_dict: Dict,
    algorithms: Sequence[str],
    epsilons: Sequence[float],
    metrics: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[int, int, str, float, str, Optional[str]], None]] = None,
) -> Iterator[Dict]:
    """Iterate over sweep combinations yielding result dictionaries.

    Args:
        cfg_dict:          Base configuration dictionary.
        algorithms:        Sequence of canonical or alias algorithm names.
        epsilons:          Sequence of epsilon values to try.
        metrics:           Optional list of evaluation metrics; defaults to
                           configuration values when omitted.
        progress_callback: Optional callable invoked with
                           ``(index, total, algorithm, epsilon, status, message)``
                           where ``status`` is one of ``"start"``, ``"ok"`` or
                           ``"failed"``.
    Yields:
        Dictionaries describing the result of each run, matching the format used
        by the Streamlit application (with ``status`` and metric columns).
    """

    cfg_metrics = resolve_metrics(cfg_dict, metrics)
    total_runs = len(algorithms) * len(epsilons)
    run_counter = 0

    for algo in algorithms:
        canonical_algo = normalise_algorithm(algo)
        if not canonical_algo:
            continue
        algo_value = algorithm_value(canonical_algo)
        for epsilon in epsilons:
            run_counter += 1
            if progress_callback:
                progress_callback(
                    run_counter,
                    total_runs,
                    canonical_algo,
                    float(epsilon),
                    "start",
                    None,
                )

            run_cfg = deepcopy(cfg_dict)
            run_cfg.setdefault("experiment", {}).pop("name", None)
            run_cfg.setdefault("semantic_compression", {})
            run_cfg["semantic_compression"]["algorithm"] = algo_value
            run_cfg["semantic_compression"]["epsilon"] = float(epsilon)
            run_cfg.setdefault("evaluation", {})["metrics"] = cfg_metrics

            try:
                result = run_experiment(run_cfg)
            except Exception as exc:  # pragma: no cover - propagation only
                message = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                if progress_callback:
                    progress_callback(
                        run_counter,
                        total_runs,
                        canonical_algo,
                        float(epsilon),
                        "failed",
                        message,
                    )
                yield {
                    "algorithm": canonical_algo,
                    "epsilon": float(epsilon),
                    "status": "failed",
                    "error": message,
                }
            else:
                metrics_dict = result.get("metrics", {})
                row: Dict[str, object] = {
                    "algorithm": canonical_algo,
                    "epsilon": float(epsilon),
                    "status": "ok",
                }
                for metric_name in cfg_metrics:
                    row[metric_name] = metrics_dict.get(metric_name)

                if progress_callback:
                    progress_callback(
                        run_counter,
                        total_runs,
                        canonical_algo,
                        float(epsilon),
                        "ok",
                        None,
                    )

                yield row


def collect_results(rows: Iterable[Dict]) -> pd.DataFrame:
    """Create a DataFrame from the sweep rows."""

    return pd.DataFrame(list(rows))


def save_results(df: pd.DataFrame, cfg_dict: Dict, output_path: str | Path | None = None) -> Path:
    cfg = Config()
    exp_cfg = cfg_dict.get("experiment", {}) or {}
    output_dir = exp_cfg.get("output_dir") or "results"
    sweep_dir = Path(cfg.project_root) / output_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if output_path is not None:
        target = Path(output_path)
        if not target.is_absolute():
            target = sweep_dir / target
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = sweep_dir / f"sweep-{timestamp}.csv"

    df.to_csv(target, index=False)
    return target


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("ClimateCompression sweep runner")
    parser.add_argument(
        "--config",
        type=str,
        default="src/ccpress/config/experiment.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="*",
        help="要 sweep 的算法名称 (使用 canonical 名称或别名)",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        help="要 sweep 的 epsilon 值",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        help="额外评估指标 (默认读取配置文件)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="将结果保存为指定 CSV (默认自动生成带时间戳的文件)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存 CSV，仅在终端输出结果",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划的 sweep 组合，不实际运行",
    )
    return parser.parse_args(argv)


def _cli_progress(step: int, total: int, algo: str, epsilon: float, status: str, message: Optional[str]):
    label = algorithm_label(algo)
    prefix = f"[{step}/{total}] {label} ε={epsilon:g}"
    if status == "start":
        print(f"→ 开始: {prefix}")
    elif status == "ok":
        print(f"✓ 完成: {prefix}")
    elif status == "failed":
        print(f"✗ 失败: {prefix}\n    {message}")


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg_dict = load_config(args.config)

    default_algos, default_eps = default_sweep_values(cfg_dict)
    algorithms = args.algorithms or default_algos
    if not algorithms:
        algorithms = list(SWEEP_ALGORITHMS.keys())
    algorithms = [algo for algo in (normalise_algorithm(a) for a in algorithms) if algo]
    algorithms = list(dict.fromkeys(algorithms))

    epsilons = args.epsilons or default_eps
    epsilons = [float(eps) for eps in epsilons]

    if not algorithms:
        print("没有可用的算法，请检查配置或命令行参数。")
        return 1

    if not epsilons:
        print("没有可用的 epsilon 值，请检查配置或命令行参数。")
        return 1

    metrics = resolve_metrics(cfg_dict, args.metrics)

    print("计划 sweep 组合:")
    for algo in algorithms:
        label = algorithm_label(algo)
        eps_list = ", ".join(f"{eps:g}" for eps in epsilons)
        print(f"  - {label}: ε ∈ [{eps_list}]")

    if args.dry_run:
        print("Dry-run 模式，未实际运行任何实验。")
        return 0

    rows = list(
        sweep_runs(
            cfg_dict,
            algorithms,
            epsilons,
            metrics=metrics,
            progress_callback=_cli_progress,
        )
    )
    df = collect_results(rows)

    if not df.empty:
        print("\nSweep 结果:")
        print(df.to_string(index=False))
        if not args.no_save:
            csv_path = save_results(df, cfg_dict, args.output)
            print(f"结果已保存到: {csv_path}")
    else:
        print("没有可用结果。")

    return 0


def main():  # pragma: no cover - CLI entry point wrapper
    raise SystemExit(run_cli())


if __name__ == "__main__":  # pragma: no cover - script execution
    main()

