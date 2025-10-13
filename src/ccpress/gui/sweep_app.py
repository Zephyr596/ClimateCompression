"""Streamlit GUI for sweeping ClimateCompression experiments."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print("[DEBUG] Added to sys.path:", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd
import streamlit as st

from ccpress.config import Config
from ccpress.main import run_experiment
from ccpress.utils.yaml_io import load_yaml


def _parse_float_list(values: Iterable[str]) -> List[float]:
    parsed: List[float] = []
    for raw in values:
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"无法解析 epsilon 值: '{raw}'") from exc
    return parsed


def _load_config(path: str | Path) -> Dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    return load_yaml(str(cfg_path))


def _default_sweep_values(cfg_dict: Dict) -> tuple[list[str], list[float]]:
    sem = cfg_dict.get("semantic_compression", {})
    sweep_cfg = cfg_dict.get("sweep", {})

    available_algorithms = []
    algo_cfg = sem.get("algorithms", {})
    if isinstance(algo_cfg, dict):
        available_algorithms = list(algo_cfg.keys())

    default_algo = sem.get("algorithm")
    if default_algo and default_algo not in available_algorithms:
        available_algorithms.append(default_algo)

    sweep_algos = sweep_cfg.get("algorithms", available_algorithms or ([default_algo] if default_algo else []))
    sweep_eps = sweep_cfg.get("epsilons", [sem.get("epsilon", 1e-3)])

    return list(dict.fromkeys(sweep_algos)), [float(e) for e in sweep_eps]


def _save_results(df: pd.DataFrame, cfg_dict: Dict) -> Path:
    cfg = Config()
    exp_cfg = cfg_dict.get("experiment", {})
    output_dir = exp_cfg.get("output_dir") or "results"
    sweep_dir = Path(cfg.project_root) / output_dir / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = sweep_dir / f"sweep-{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    st.set_page_config(page_title="ClimateCompression Sweep", layout="wide")
    st.title("ClimateCompression 实验 Sweep")

    st.sidebar.header("配置")
    default_cfg_path = "src/ccpress/config/experiment.yaml"
    cfg_path = st.sidebar.text_input("配置文件路径", value=default_cfg_path)

    try:
        cfg_dict = _load_config(cfg_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    sem = cfg_dict.get("semantic_compression", {})
    algorithms_cfg = sem.get("algorithms", {}) if isinstance(sem, dict) else {}
    available_algorithms = sorted(algorithms_cfg.keys())
    default_algorithms, default_eps = _default_sweep_values(cfg_dict)

    selected_algorithms = st.sidebar.multiselect(
        "选择压缩算法",
        options=available_algorithms or default_algorithms,
        default=default_algorithms or available_algorithms,
        help="从配置中可用的压缩算法中进行选择",
    )

    epsilon_text = st.sidebar.text_area(
        "Epsilon 值 (用逗号/空格分隔)",
        value="\n".join([", ".join(str(e) for e in default_eps)]),
        height=80,
    )

    metric_options = ["compression_ratio", "mse", "psnr"]
    default_metrics = cfg_dict.get("evaluation", {}).get("metrics", ["compression_ratio"])
    selected_metrics = st.sidebar.multiselect(
        "评估指标",
        options=metric_options,
        default=[m for m in default_metrics if m in metric_options] or ["compression_ratio"],
    )

    run_button = st.sidebar.button("运行 Sweep", type="primary")

    st.write("### 当前基础配置")
    st.json(cfg_dict)

    if not run_button:
        st.info("在侧边栏配置好参数后点击“运行 Sweep”开始实验。")
        return

    try:
        epsilon_values = _parse_float_list(epsilon_text.replace("\n", ",").split(","))
    except ValueError as exc:
        st.error(str(exc))
        return

    if not selected_algorithms:
        st.warning("请至少选择一个压缩算法。")
        return

    if not epsilon_values:
        st.warning("请至少提供一个 epsilon 值。")
        return

    total_runs = len(selected_algorithms) * len(epsilon_values)
    progress = st.progress(0.0)
    results: List[Dict] = []
    log_container = st.empty()

    for idx, algo in enumerate(selected_algorithms):
        for jdx, epsilon in enumerate(epsilon_values):
            run_idx = idx * len(epsilon_values) + jdx
            progress.progress((run_idx) / total_runs)

            run_cfg = deepcopy(cfg_dict)
            run_cfg.setdefault("experiment", {}).pop("name", None)
            run_cfg["semantic_compression"]["algorithm"] = algo
            run_cfg["semantic_compression"]["epsilon"] = float(epsilon)
            run_cfg.setdefault("evaluation", {})["metrics"] = selected_metrics

            log_container.info(f"运行 {algo} @ epsilon={epsilon} ({run_idx + 1}/{total_runs}) ...")

            try:
                result = run_experiment(run_cfg)
            except Exception as exc:
                import traceback
                err_msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                st.error(f"{algo}@epsilon={epsilon} 失败：\n```\n{err_msg}\n```")
                results.append({
                    "algorithm": algo,
                    "epsilon": epsilon,
                    "status": "failed",
                    "error": str(exc),
                })

            else:
                metrics = result.get("metrics", {})
                row = {
                    "algorithm": algo,
                    "epsilon": epsilon,
                    "status": "ok",
                }
                for metric_name in selected_metrics:
                    row[metric_name] = metrics.get(metric_name)
                results.append(row)

            progress.progress((run_idx + 1) / total_runs)

    progress.empty()
    log_container.empty()

    if not results:
        st.warning("没有可显示的结果。")
        return

    df = pd.DataFrame(results)
    st.write("### Sweep 结果")
    st.dataframe(df)

    if "compression_ratio" in df.columns and df["compression_ratio"].notna().any():
        chart = (
            pd.melt(
                df[df["status"] == "ok"],
                id_vars=["algorithm", "epsilon"],
                value_vars=["compression_ratio"],
            )
            .rename(columns={"variable": "metric", "value": "value"})
        )
        if not chart.empty:
            import altair as alt

            st.write("#### Compression Ratio 对比")
            st.altair_chart(
                alt.Chart(chart)
                .mark_line(point=True)
                .encode(
                    x=alt.X("epsilon:Q", scale=alt.Scale(type="log" if chart["epsilon"].min() > 0 else "linear")),
                    y="value:Q",
                    color="algorithm:N",
                )
                .properties(height=400),
                use_container_width=True,
            )

    csv_path = _save_results(df, cfg_dict)
    st.success(f"Sweep 结果已保存至 {csv_path}")


if __name__ == "__main__":
    main()
