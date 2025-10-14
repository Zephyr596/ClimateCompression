"""Streamlit GUI for sweeping ClimateCompression experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import sys

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from ccpress.sweep import (
    SWEEP_ALGORITHMS,
    algorithm_label,
    collect_results,
    default_sweep_values,
    load_config,
    normalise_algorithm,
    save_results,
    sweep_runs,
)


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


def main():
    st.set_page_config(page_title="ClimateCompression Sweep", layout="wide")
    st.title("ClimateCompression 实验 Sweep")

    st.sidebar.header("配置")
    default_cfg_path = "src/ccpress/config/experiment.yaml"
    cfg_path = st.sidebar.text_input("配置文件路径", value=default_cfg_path)

    try:
        cfg_dict = load_config(cfg_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    sem = cfg_dict.get("semantic_compression", {}) or {}
    algorithms_cfg = sem.get("algorithms", {}) if isinstance(sem, dict) else {}

    available_algorithms = []
    for name in algorithms_cfg.keys():
        norm = normalise_algorithm(name)
        if norm and norm not in available_algorithms:
            available_algorithms.append(norm)

    default_algorithms, default_eps = default_sweep_values(cfg_dict)

    all_algorithms: list[str] = []
    for source in (default_algorithms, available_algorithms, list(SWEEP_ALGORITHMS.keys())):
        for algo in source:
            if algo and algo not in all_algorithms:
                all_algorithms.append(algo)

    selected_algorithms = st.sidebar.multiselect(
        "选择压缩算法",
        options=all_algorithms,
        default=default_algorithms or all_algorithms,
        format_func=lambda key: algorithm_label(key),
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
    log_container = st.empty()
    metrics = [m.lower() for m in selected_metrics]

    def _update_progress(step: int, total: int, algo: str, epsilon: float, status: str, message: str | None):
        progress.progress(step / total)
        if status == "start":
            log_container.info(
                f"运行 {algorithm_label(algo)} @ epsilon={epsilon} ({step}/{total}) ..."
            )
        elif status == "failed" and message:
            log_container.error(
                f"{algorithm_label(algo)}@epsilon={epsilon} 失败:\n``\n{message}\n```"
            )

    rows = list(
        sweep_runs(
            cfg_dict,
            selected_algorithms,
            epsilon_values,
            metrics=metrics,
            progress_callback=_update_progress,
        )
    )

    progress.empty()
    log_container.empty()

    if not rows:
        st.warning("没有可显示的结果。")
        return

    df = collect_results(rows)
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

    csv_path = save_results(df, cfg_dict)
    st.success(f"Sweep 结果已保存至 {csv_path}")


if __name__ == "__main__":
    main()
