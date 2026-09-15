"""Validate trained CMC experiments and create publication-ready evidence.

This script is intentionally strict: the default analysis is produced only when
every preregistered dataset/model/seed cell exists, all evaluations are full-test
runs, and each dataset uses one immutable test split and threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import PALETTE, add_panel_labels, export_figure, publication_subplots  # noqa: E402


DATASETS = ("nudt", "irstd1k", "sirstaug")
MODELS = (
    "rirufold_admm",
    "rirufold_rcp",
    "rirufold_cmc_nomask",
    "rirufold_cmc_noside",
    "rirufold_cmc_onepass",
    "rirufold_cmc_fixedmask",
    "rirufold_cmc",
)
MAIN_MODELS = ("rirufold_admm", "rirufold_rcp", "rirufold_cmc")
ABLATION_MODELS = (
    "rirufold_cmc",
    "rirufold_cmc_nomask",
    "rirufold_cmc_noside",
    "rirufold_cmc_onepass",
    "rirufold_cmc_fixedmask",
)
MODEL_LABELS = {
    "rirufold_admm": "ADMM",
    "rirufold_rcp": "RCP",
    "rirufold_cmc": "CMC",
    "rirufold_cmc_nomask": "No mask",
    "rirufold_cmc_noside": "No side",
    "rirufold_cmc_onepass": "One pass",
    "rirufold_cmc_fixedmask": "Fixed mask",
}
METRICS = ("mIoU", "F1", "Pd", "Fa_1e6_per_pixel")
METRIC_LABELS = {
    "mIoU": "mIoU (%)",
    "F1": "F1 (%)",
    "Pd": "Pd (%)",
    "Fa_1e6_per_pixel": "Fa per 10^6 px",
}
REQUIRED_KEYS = {
    "dataset", "net_name", "seed", "threshold", "partial_evaluation",
    "test_images", "full_test_images", "checkpoint_sha256", "test_split_sha256",
    "mIoU", "precision", "recall", "F1", "Pd",
    "Fa_1e6_per_pixel", "seconds_per_image", "trainable_parameters",
}
COLORS = (PALETTE["neutral"], PALETTE["secondary"], PALETTE["primary"],
          PALETTE["contrast"], PALETTE["positive"])
MARKERS = ("o", "s", "^", "D", "v")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict paired analysis for trained CMC-RiRUFold experiments"
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2026])
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    return parser.parse_args()


def load_records(input_root: Path) -> pd.DataFrame:
    paths = sorted(input_root.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON result files found under {input_root}")
    records = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(REQUIRED_KEYS.difference(record))
        if missing:
            raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
        record["source_json"] = str(path.resolve())
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    duplicate = frame.duplicated(["dataset", "net_name", "seed"], keep=False)
    if duplicate.any():
        cells = frame.loc[duplicate, ["dataset", "net_name", "seed", "source_json"]]
        raise ValueError("Duplicate experiment cells:\n" + cells.to_string(index=False))
    return frame


def validate_matrix(frame: pd.DataFrame, seeds: Iterable[int]) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [DATASETS, MODELS, tuple(seeds)], names=["dataset", "net_name", "seed"]
    )
    actual = pd.MultiIndex.from_frame(frame[["dataset", "net_name", "seed"]])
    unexpected = actual.difference(expected)
    missing = expected.difference(actual)
    if len(unexpected):
        raise ValueError(f"Unexpected dataset/model/seed cells: {list(unexpected)}")
    if len(missing):
        raise ValueError(f"Incomplete preregistered matrix; missing: {list(missing)}")
    if frame["partial_evaluation"].astype(bool).any():
        bad = frame.loc[frame["partial_evaluation"].astype(bool), "source_json"].tolist()
        raise ValueError(f"Partial evaluations cannot enter the paper analysis: {bad}")
    incomplete_count = frame["test_images"].astype(int) != frame["full_test_images"].astype(int)
    if incomplete_count.any():
        bad = frame.loc[incomplete_count, "source_json"].tolist()
        raise ValueError(f"test_images does not equal full_test_images: {bad}")
    checkpoint_hash = frame["checkpoint_sha256"].astype(str).str.upper()
    if not checkpoint_hash.str.fullmatch(r"[0-9A-F]{64}").all():
        raise ValueError("Every experiment must contain a valid checkpoint SHA-256")
    reused = checkpoint_hash.duplicated(keep=False)
    if reused.any():
        cells = frame.loc[reused, ["dataset", "net_name", "seed", "checkpoint_sha256"]]
        raise ValueError("The same checkpoint was reused by multiple experiment cells:\n" + cells.to_string(index=False))
    for dataset, group in frame.groupby("dataset"):
        if group["test_split_sha256"].nunique() != 1:
            raise ValueError(f"{dataset} has inconsistent test split hashes")
        if group["threshold"].nunique() != 1:
            raise ValueError(f"{dataset} has inconsistent decision thresholds")
    numeric = [*METRICS, "precision", "recall", "seconds_per_image", "trainable_parameters"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Metrics contain NaN or infinity")
    return frame.sort_values(["dataset", "net_name", "seed"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [*METRICS, "precision", "recall", "seconds_per_image", "trainable_parameters"]
    grouped = frame.groupby(["dataset", "net_name"], sort=True)
    output = grouped[numeric].agg(["mean", "std"])
    output.columns = [f"{metric}_{stat}" for metric, stat in output.columns]
    output.insert(0, "seed_count", grouped["seed"].nunique())
    return output.reset_index()


def paired_gains(
    frame: pd.DataFrame,
    repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    if repeats < 1000:
        raise ValueError("bootstrap-repeats must be at least 1000")
    rng = np.random.default_rng(random_seed)
    rows = []
    comparators = [name for name in MODELS if name != "rirufold_cmc"]
    for dataset in DATASETS:
        subset = frame[frame["dataset"] == dataset]
        cmc = subset[subset["net_name"] == "rirufold_cmc"].set_index("seed")
        for comparator in comparators:
            other = subset[subset["net_name"] == comparator].set_index("seed")
            common = cmc.index.intersection(other.index).sort_values()
            for metric in METRICS:
                cmc_values = cmc.loc[common, metric].to_numpy(dtype=float)
                other_values = other.loc[common, metric].to_numpy(dtype=float)
                difference = (
                    other_values - cmc_values
                    if metric == "Fa_1e6_per_pixel"
                    else cmc_values - other_values
                )
                indices = rng.integers(0, len(difference), size=(repeats, len(difference)))
                boot = difference[indices].mean(axis=1)
                rows.append({
                    "dataset": dataset,
                    "comparator": comparator,
                    "metric": metric,
                    "paired_seed_count": len(difference),
                    "gain_mean": float(difference.mean()),
                    "gain_std": float(difference.std(ddof=1)),
                    "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                    "gain_definition": "comparator-CMC" if metric == "Fa_1e6_per_pixel" else "CMC-comparator",
                })
    return pd.DataFrame.from_records(rows)


def _metric_values(summary: pd.DataFrame, model: str, metric: str):
    indexed = summary.set_index(["dataset", "net_name"])
    mean = np.array([indexed.loc[(dataset, model), f"{metric}_mean"] for dataset in DATASETS])
    std = np.array([indexed.loc[(dataset, model), f"{metric}_std"] for dataset in DATASETS])
    scale = 100.0 if metric != "Fa_1e6_per_pixel" else 1.0
    return mean * scale, std * scale


def plot_comparison(summary: pd.DataFrame, models: tuple[str, ...], stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = publication_subplots(2, 2, width="double", aspect=0.72, squeeze=False)
    x = np.arange(len(DATASETS))
    for axis, metric in zip(axes.flat, METRICS):
        for index, model in enumerate(models):
            mean, std = _metric_values(summary, model, metric)
            axis.errorbar(
                x, mean, yerr=std, label=MODEL_LABELS[model], color=COLORS[index],
                marker=MARKERS[index], linestyle=LINESTYLES[index], capsize=2,
            )
        axis.set(title=METRIC_LABELS[metric], xticks=x, xticklabels=("NUDT", "IRSTD-1K", "SIRST-Aug"))
        axis.set_ylabel(METRIC_LABELS[metric])
    add_panel_labels(axes.flat)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(models))
    export_figure(fig, stem)
    plt.close(fig)


def plot_paired_rcp_gain(gains: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt

    selected = gains[gains["comparator"] == "rirufold_rcp"]
    fig, axes = publication_subplots(2, 2, width="double", aspect=0.72, squeeze=False)
    y = np.arange(len(DATASETS))
    for axis, metric in zip(axes.flat, METRICS):
        rows = selected[selected["metric"] == metric].set_index("dataset").loc[list(DATASETS)]
        scale = 100.0 if metric != "Fa_1e6_per_pixel" else 1.0
        mean = rows["gain_mean"].to_numpy() * scale
        low = rows["bootstrap_ci95_low"].to_numpy() * scale
        high = rows["bootstrap_ci95_high"].to_numpy() * scale
        axis.errorbar(
            mean, y, xerr=np.vstack((mean - low, high - mean)), fmt="o",
            color=PALETTE["primary"], capsize=2,
        )
        axis.axvline(0.0, color=PALETTE["dark"], linewidth=0.7, linestyle="--")
        unit = "percentage points" if metric != "Fa_1e6_per_pixel" else "per 10^6 px"
        axis.set(title=METRIC_LABELS[metric], xlabel=f"CMC gain vs RCP ({unit})", yticks=y,
                 yticklabels=("NUDT", "IRSTD-1K", "SIRST-Aug"))
    add_panel_labels(axes.flat)
    fig.suptitle("Positive values favor CMC; paired bootstrap is descriptive for n=3 seeds", fontsize=8)
    export_figure(fig, stem)
    plt.close(fig)


def write_report(summary: pd.DataFrame, gains: pd.DataFrame, path: Path, seeds: list[int]) -> None:
    indexed = summary.set_index(["dataset", "net_name"])
    paired = gains[gains["comparator"] == "rirufold_rcp"].set_index(["dataset", "metric"])
    lines = [
        "# CMC-RiRUFold 服务器训练结果报告",
        "",
        "> 本文件由完整服务器评测 JSON 自动生成。只有通过完整性、重复单元、partial run、",
        "> split hash 和 threshold 一致性检查后才会写出。配对 bootstrap 在每数据集仅 3 个",
        "> seed 时只作描述性区间，不替代更多重复实验。",
        "",
        f"- 种子：{', '.join(map(str, seeds))}",
        f"- 实验单元：{len(DATASETS)} 数据集 × {len(MODELS)} 模型 × {len(seeds)} 种子",
        "- 增益方向：mIoU/F1/Pd 为 `CMC-RCP`；Fa 为 `RCP-CMC`，因此均为正值更好。",
        "",
        "## CMC 主结果（mean ± std）",
        "",
        "| 数据集 | mIoU | F1 | Pd | Fa/10^6 px |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        row = indexed.loc[(dataset, "rirufold_cmc")]
        cells = []
        for metric in METRICS:
            scale = 100.0 if metric != "Fa_1e6_per_pixel" else 1.0
            cells.append(f"{row[f'{metric}_mean']*scale:.3f} ± {row[f'{metric}_std']*scale:.3f}")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines += ["", "## CMC 相对 RCP 的配对增益", "",
              "| 数据集 | 指标 | 平均增益 | 95% bootstrap CI |", "|---|---|---:|---:|"]
    for dataset in DATASETS:
        for metric in METRICS:
            row = paired.loc[(dataset, metric)]
            scale = 100.0 if metric != "Fa_1e6_per_pixel" else 1.0
            lines.append(
                f"| {dataset} | {METRIC_LABELS[metric]} | {row['gain_mean']*scale:.4f} | "
                f"[{row['bootstrap_ci95_low']*scale:.4f}, {row['bootstrap_ci95_high']*scale:.4f}] |"
            )
    lines += [
        "", "## 结论使用规则", "",
        "- 只有至少 2/3 数据集的三种子平均 mIoU 增益达到 0.30 个百分点、Fa 相对增幅",
        "  不超过 5%，且目标区背景吸收率下降，CMC 才进入论文主表。",
        "- 若 `onepass` 不劣于 CMC，不把两步交换单列为有效贡献。",
        "- 若 `noside` 与 CMC 无差异，撤回侧先验贡献主张。",
        "- 本报告只读取评测输出，不自动把统计差异表述为统计显著或因果效果。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = validate_matrix(load_records(args.input_root), args.seeds)
    summary = summarize(frame)
    gains = paired_gains(frame, args.bootstrap_repeats, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "server_seed_records.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "server_summary.csv", index=False, encoding="utf-8-sig")
    gains.to_csv(args.output_dir / "server_paired_gains.csv", index=False, encoding="utf-8-sig")
    plot_comparison(summary, MAIN_MODELS, args.output_dir / "result_q2_server_main")
    plot_comparison(summary, ABLATION_MODELS, args.output_dir / "result_q2_server_ablation")
    plot_paired_rcp_gain(gains, args.output_dir / "result_q2_server_paired_gain")
    write_report(summary, gains, args.output_dir / "CMC服务器结果报告.md", args.seeds)
    print(f"Validated {len(frame)} complete experiment cells; wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
