"""Run deterministic mechanism experiments and produce publication candidates.

The outputs are explicitly *not* trained benchmark results.  They test whether
the proposed state chain, constraints, diagnostics, and optional blob branch
behave in the intended direction before expensive server training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.ndimage as ndi  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


_RESAMPLING = getattr(Image, "Resampling", Image)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RiRUFold_ADMM.models.rirufold_rcp import build_rcp_model  # noqa: E402
from utils.plot_style import (  # noqa: E402
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)


VARIANT_LABELS = {
    "main": "RCP main",
    "product": "Fixed product",
    "blob": "RCP-Blob",
}
VARIANT_COLORS = {
    "main": PALETTE["primary"],
    "product": PALETTE["neutral"],
    "blob": PALETTE["contrast"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic RCP-RiRUFold mechanism study")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--stage-num", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--scene-size", type=int, default=40)
    parser.add_argument("--mechanism-scenes", type=int, default=8)
    parser.add_argument("--profile-images", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_scene(index: int, size: int, with_target: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9301 + index)
    yy, xx = np.mgrid[0:size, 0:size]
    edge_x = size // 2 + (index % 3 - 1) * 3
    slope = 0.35 + 0.08 * (index % 4)
    ridge = np.exp(-0.5 * ((yy - slope * xx - size * 0.18) / 1.15) ** 2)
    background = (
        0.10
        + 0.0014 * xx
        + 0.0008 * yy
        + 0.12 * (xx >= edge_x)
        + 0.035 * ridge
        + 0.014 * np.sin((xx + 1.3 * yy) * (0.16 + 0.01 * index))
        + rng.normal(0.0, 0.005, size=(size, size))
    )
    target = np.zeros((size, size), dtype=bool)
    if with_target:
        center_x = edge_x - 3 + (index % 5)
        center_y = size // 3 + (index * 5) % max(4, size // 3)
        radius = 1.15 + 0.12 * (index % 3)
        blob = np.exp(-0.5 * (((xx - center_x) / radius) ** 2 + ((yy - center_y) / radius) ** 2))
        background = background + (0.34 + 0.03 * (index % 4)) * blob
        target = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= 2.25 * radius**2
    edge = (np.abs(xx - edge_x) <= 2) | (np.abs(yy - slope * xx - size * 0.18) <= 1.5)
    edge &= ~target
    return np.clip(background, 0.0, 1.0).astype(np.float32), target, edge


def as_tensor(arrays: Iterable[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(list(arrays), axis=0)[:, None].astype(np.float32))


def profile_nudt(profile_images: int) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, List[Path]]:
    image_dir = PROJECT_ROOT / "datasets" / "NUDT-SIRST" / "test" / "images"
    mask_dir = PROJECT_ROOT / "datasets" / "NUDT-SIRST" / "test" / "masks"
    image_paths = sorted(image_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not image_paths or len(image_paths) != len(mask_paths):
        raise RuntimeError("NUDT test image/mask pairs are missing or mismatched")

    selected_names = {path.name for path in image_paths[:profile_images]}
    intensity_samples: List[np.ndarray] = []
    component_areas: List[int] = []
    rows: List[Dict[str, object]] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        labeled, component_count = ndi.label(mask, structure=structure)
        areas = np.bincount(labeled.ravel())[1:].astype(int)
        component_areas.extend(areas.tolist())
        rows.append(
            {
                "file": image_path.name,
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
                "height_px": image.shape[0],
                "width_px": image.shape[1],
                "intensity_min": float(image.min()),
                "intensity_max": float(image.max()),
                "intensity_mean": float(image.mean()),
                "intensity_std": float(image.std()),
                "target_pixels": int(mask.sum()),
                "target_components_8conn": int(component_count),
            }
        )
        if image_path.name in selected_names:
            intensity_samples.append(image.ravel()[::8])
    return rows, np.concatenate(intensity_samples), np.asarray(component_areas), image_paths


def make_models(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    main = build_rcp_model(
        "rirufold_rcp", stage_num=args.stage_num, hidden_channels=args.hidden_channels
    ).eval()
    product = build_rcp_model(
        "rirufold_rcp_product", stage_num=args.stage_num, hidden_channels=args.hidden_channels
    ).eval()
    blob = build_rcp_model(
        "rirufold_rcp_blob", stage_num=args.stage_num, hidden_channels=args.hidden_channels
    ).eval()
    product.load_state_dict(main.state_dict())
    blob.load_state_dict(main.state_dict())
    return main, product, blob


def set_blob_gamma(model: torch.nn.Module, gamma: float) -> None:
    if not 0.0 < gamma < 0.5:
        raise ValueError("blob gamma must be inside (0, 0.5)")
    raw = math.log((gamma / 0.5) / (1.0 - gamma / 0.5))
    with torch.no_grad():
        for stage in model.stages:
            stage.weight_model.raw_blob_gamma.fill_(raw)


def run_with_trace(model: torch.nn.Module, image: torch.Tensor):
    with torch.no_grad():
        background, logits, trace = model(image, return_trace=True)
        _, _, aux = model(image, return_aux=True)
    return background, logits, aux["sparse_intensity"].detach(), trace


def scene_rows(
    variant: str,
    sparse: torch.Tensor,
    trace: List[Dict[str, torch.Tensor]],
    masks: np.ndarray,
    edges: np.ndarray,
    empty_sparse: torch.Tensor,
    empty_reference_threshold: float,
) -> List[Dict[str, object]]:
    threshold = trace[-1]["sparse_threshold"].cpu().numpy()[:, 0]
    sparse_np = sparse.cpu().numpy()[:, 0]
    empty_np = empty_sparse.cpu().numpy()[:, 0]
    rows = []
    for index in range(sparse_np.shape[0]):
        target_response = float(sparse_np[index][masks[index]].mean())
        edge_response = float(sparse_np[index][edges[index]].mean())
        rows.append(
            {
                "scene": index,
                "variant": variant,
                "target_sparse_mean": target_response,
                "edge_sparse_mean": edge_response,
                "target_edge_ratio": target_response / max(edge_response, 1.0e-4),
                "target_edge_contrast": target_response - edge_response,
                "target_threshold_mean": float(threshold[index][masks[index]].mean()),
                "edge_threshold_mean": float(threshold[index][edges[index]].mean()),
                "empty_exceedance_fraction": float(
                    (empty_np[index] > empty_reference_threshold).mean()
                ),
            }
        )
    return rows


def aggregate(rows: List[Dict[str, object]], variant: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["variant"] == variant]
    return float(np.mean(values))


def median_metric(rows: List[Dict[str, object]], variant: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["variant"] == variant]
    return float(np.median(values))


def hide_image_axis(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def export_raw_figures(
    figures_dir: Path,
    intensity_samples: np.ndarray,
    component_areas: np.ndarray,
    scene: np.ndarray,
    target: np.ndarray,
    edge: np.ndarray,
) -> None:
    values = np.sort(intensity_samples)
    ecdf = np.arange(1, values.size + 1) / values.size
    fig, axis = publication_subplots(width="report", aspect=0.55)
    axis.plot(values, ecdf, color=PALETTE["primary"])
    axis.set(title="NUDT intensity ECDF", xlabel="Normalized infrared intensity", ylabel="Empirical cumulative fraction")
    axis.text(0.98, 0.05, "32 images; full pixels sampled every 8", transform=axis.transAxes, ha="right", color=PALETTE["neutral"])
    export_figure(fig, figures_dir / "raw_q1_nudt_intensity_ecdf")
    plt.close(fig)

    fig, axes = publication_subplots(1, 2, width="report", aspect=0.52, width_ratios=[1.55, 1.0])
    positive_areas = component_areas[component_areas > 0]
    bins = np.geomspace(1, max(2, positive_areas.max()), 18)
    axes[0].hist(positive_areas, bins=bins, color=PALETTE["primary"], alpha=0.85)
    axes[0].set_xscale("log")
    axes[0].set(title="Component areas", xlabel="8-connected area (pixel)", ylabel="Target count")
    sorted_areas = np.sort(positive_areas)
    axes[1].plot(sorted_areas, np.arange(1, len(sorted_areas) + 1) / len(sorted_areas), color=PALETTE["contrast"])
    axes[1].axvline(np.median(sorted_areas), color=PALETTE["dark"], linestyle="--", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set(title="Area ECDF", xlabel="8-connected area (pixel)", ylabel="Cumulative fraction")
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "raw_q1_target_area_distribution")
    plt.close(fig)

    fig, axes = publication_subplots(1, 3, width="report", aspect=0.37)
    image_artist = axes[0].imshow(scene, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Synthetic input")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Target mask")
    axes[2].imshow(edge, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Clutter edge")
    for axis in axes:
        hide_image_axis(axis)
    fig.colorbar(image_artist, ax=axes[0], label="Normalized intensity", fraction=0.047)
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "raw_q1_synthetic_scene")
    plt.close(fig)


def export_process_figures(
    figures_dir: Path,
    main_trace: List[Dict[str, torch.Tensor]],
    traces: Dict[str, List[Dict[str, torch.Tensor]]],
    target_row: int,
) -> None:
    stages = np.arange(1, len(main_trace) + 1)
    primal = np.asarray([float(item["primal_norm"].mean()) for item in main_trace])
    dual = np.asarray([float(item["dual_norm"].mean()) for item in main_trace])
    mu = np.asarray([float(item["mu_next"].mean()) for item in main_trace])
    fig, axis = publication_subplots(width="report", aspect=0.55)
    axis.plot(stages, primal / max(primal[0], 1e-8), marker="o", color=PALETTE["primary"], label="Primal norm")
    axis.plot(stages, dual / max(dual[0], 1e-8), marker="s", color=PALETTE["contrast"], label="Dual proxy")
    axis.plot(stages, mu / max(mu[0], 1e-8), marker="^", color=PALETTE["positive"], label="Penalty mu")
    axis.axhline(1.0, color=PALETTE["neutral"], linestyle=":", linewidth=0.8)
    axis.set(title="Stage-wise state trajectory", xlabel="Unfolding stage", ylabel="Value / stage-1 value", xticks=stages)
    axis.legend(ncols=3, loc="upper center")
    export_figure(fig, figures_dir / "process_q1_residual_trajectory")
    plt.close(fig)

    branch_means = np.asarray(
        [item["branch_weights"].mean(dim=(0, 2, 3, 4)).cpu().numpy() for item in main_trace]
    )
    uncertainty = main_trace[-1]["uncertainty"].cpu().numpy().ravel()
    fig, axes = publication_subplots(1, 2, width="report", aspect=0.52, width_ratios=[1.45, 1.0])
    labels = ["8/4", "12/6", "16/8"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["positive"]]
    markers = ["o", "s", "^"]
    for index, label in enumerate(labels):
        axes[0].plot(stages, branch_means[:, index], marker=markers[index], color=colors[index], label=label)
    axes[0].set(title="Patch consensus", xlabel="Unfolding stage", ylabel="Mean branch weight", xticks=stages, ylim=(0, 1))
    axes[0].legend(title="Patch/stride", ncols=3)
    axes[1].hist(uncertainty, bins=18, color=PALETTE["neutral"], alpha=0.85)
    axes[1].set(title="Final disagreement", xlabel="Normalized disagreement", ylabel="Pixel count")
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "process_q1_patch_consensus")
    plt.close(fig)

    fig, axis = publication_subplots(width="report", aspect=0.52)
    x = np.arange(traces["main"][-1]["sparse_threshold"].shape[-1])
    line_styles = {"main": "-", "product": "--", "blob": "-."}
    for variant, trace in traces.items():
        profile = trace[-1]["sparse_threshold"][0, 0, target_row].cpu().numpy()
        axis.plot(
            x,
            profile,
            color=VARIANT_COLORS[variant],
            linestyle=line_styles[variant],
            label=VARIANT_LABELS[variant],
        )
    axis.set(title="Threshold through target row", xlabel="Horizontal pixel", ylabel="Sparse threshold (normalized intensity)")
    axis.legend(ncols=3, loc="upper center")
    export_figure(fig, figures_dir / "process_q1_threshold_profile")
    plt.close(fig)


def export_result_figures(
    figures_dir: Path,
    scene: np.ndarray,
    target: np.ndarray,
    main_output,
    metric_rows: List[Dict[str, object]],
    real_image: torch.Tensor,
    real_output,
    main_edge_gate: float,
    blob_ratio_gate: float,
    blob_scan_rows: List[Dict[str, object]],
) -> None:
    background, logits, sparse, _ = main_output
    background_np = background[0, 0].cpu().numpy()
    sparse_np = sparse[0, 0].cpu().numpy()
    probability_np = torch.sigmoid(logits)[0, 0].cpu().numpy()
    fig, axes = publication_subplots(1, 4, width="double", aspect=0.31)
    im0 = axes[0].imshow(scene, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[1].imshow(background_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Background B")
    im2 = axes[2].imshow(sparse_np, cmap="magma", vmin=0, vmax=max(float(sparse_np.max()), 1e-5))
    axes[2].set_title("Sparse intensity S")
    im3 = axes[3].imshow(probability_np, cmap="viridis", vmin=0, vmax=1)
    axes[3].contour(target.astype(float), levels=[0.5], colors=["white"], linewidths=0.55)
    axes[3].set_title("Probability")
    for axis in axes:
        hide_image_axis(axis)
    fig.colorbar(im0, ax=axes[:2], label="Normalized intensity", fraction=0.027)
    fig.colorbar(im2, ax=axes[2], label="Sparse intensity", fraction=0.047)
    fig.colorbar(im3, ax=axes[3], label="Probability", fraction=0.047)
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "result_q1_synthetic_decomposition")
    plt.close(fig)

    variants = ["product", "main", "blob"]
    fig, axes = publication_subplots(1, 2, width="report", aspect=0.54, width_ratios=[1.0, 1.0])
    for axis, field, title, gate in (
        (axes[0], "edge_sparse_mean", "Edge leakage", main_edge_gate),
        (axes[1], "target_edge_ratio", "Target / edge response", blob_ratio_gate),
    ):
        for position, variant in enumerate(variants):
            values = np.asarray([float(row[field]) for row in metric_rows if row["variant"] == variant])
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(np.full(len(values), position) + jitter, values, s=11, color=VARIANT_COLORS[variant], alpha=0.72)
            axis.plot([position - 0.17, position + 0.17], [np.median(values)] * 2, color=PALETTE["dark"], linewidth=1.1)
        axis.axhline(gate, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
        axis.set(title=title, xticks=range(3), xticklabels=[VARIANT_LABELS[item] for item in variants])
        axis.tick_params(axis="x", rotation=18)
    axes[0].set_ylabel("Mean sparse response (normalized intensity)")
    axes[1].set_ylabel("Response ratio")
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "result_q1_mechanism_metrics")
    plt.close(fig)

    fig, axis = publication_subplots(width="report", aspect=0.55)
    false_alarm_increase = np.asarray(
        [100.0 * float(row["empty_fraction_increase"]) for row in blob_scan_rows]
    )
    target_response = np.asarray(
        [float(row["median_target_sparse"]) for row in blob_scan_rows]
    )
    axis.scatter(false_alarm_increase, target_response, s=28, color=PALETTE["contrast"])
    for x_value, y_value, row in zip(false_alarm_increase, target_response, blob_scan_rows):
        axis.annotate(
            "g={:.2f}".format(float(row["blob_gamma"])),
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axis.axvline(0.1, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    axis.axhline(5.0e-4, color=PALETTE["neutral"], linestyle=":", linewidth=0.8)
    axis.set(
        title="Blob trade-off",
        xlabel="Empty-scene exceedance increase (percentage point)",
        ylabel="Median target sparse response",
    )
    export_figure(fig, figures_dir / "result_q1_blob_tradeoff")
    plt.close(fig)

    real_background, real_logits, real_sparse, real_trace = real_output
    real_np = real_image[0, 0].cpu().numpy()
    real_b = real_background[0, 0].cpu().numpy()
    real_s = real_sparse[0, 0].cpu().numpy()
    real_w = real_trace[-1]["w_rcp"][0, 0].cpu().numpy()
    fig, axes = publication_subplots(1, 4, width="double", aspect=0.31)
    im0 = axes[0].imshow(real_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("NUDT input")
    axes[1].imshow(real_b, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Background B")
    im2 = axes[2].imshow(real_s, cmap="magma", vmin=0, vmax=max(float(real_s.max()), 1e-5))
    axes[2].set_title("Sparse intensity S")
    im3 = axes[3].imshow(real_w, cmap="viridis", vmin=0.25, vmax=4.0)
    axes[3].set_title("Penalty weight W")
    for axis in axes:
        hide_image_axis(axis)
    fig.colorbar(im0, ax=axes[:2], label="Normalized intensity", fraction=0.027)
    fig.colorbar(im2, ax=axes[2], label="Sparse intensity", fraction=0.047)
    fig.colorbar(im3, ax=axes[3], label="Dimensionless weight", fraction=0.047)
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "result_q1_real_nudt_trace")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.mechanism_scenes < 3:
        raise ValueError("mechanism-scenes must be at least 3")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    apply_publication_style(language="en", width="report")

    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    profile_rows, intensity_samples, component_areas, image_paths = profile_nudt(args.profile_images)
    write_csv(results_dir / "nudt_test_profile.csv", profile_rows)
    first_scene, first_target, first_edge = synthetic_scene(0, args.scene_size, with_target=True)
    export_raw_figures(
        figures_dir, intensity_samples, component_areas, first_scene, first_target, first_edge
    )

    scenes, masks, edges = zip(
        *(synthetic_scene(index, args.scene_size, with_target=True) for index in range(args.mechanism_scenes))
    )
    empty_scenes, _, empty_edges = zip(
        *(synthetic_scene(index, args.scene_size, with_target=False) for index in range(args.mechanism_scenes))
    )
    image_batch = as_tensor(scenes)
    empty_batch = as_tensor(empty_scenes)
    mask_array = np.stack(masks)
    edge_array = np.stack(edges)

    main_model, product_model, blob_model = make_models(args)
    start = time.perf_counter()
    main_background, main_logits, main_sparse, main_trace = run_with_trace(main_model, image_batch)
    main_empty = run_with_trace(main_model, empty_batch)[2]
    product_background, product_logits, product_sparse, product_trace = run_with_trace(product_model, image_batch)
    product_empty = run_with_trace(product_model, empty_batch)[2]
    reference_threshold = max(float(torch.quantile(main_empty.flatten(), 0.995)), 1.0e-8)

    main_rows = scene_rows(
        "main", main_sparse, main_trace, mask_array, edge_array, main_empty, reference_threshold
    )
    product_rows = scene_rows(
        "product", product_sparse, product_trace, mask_array, edge_array, product_empty, reference_threshold
    )
    main_ratio = median_metric(main_rows, "main", "target_edge_ratio")
    main_target = median_metric(main_rows, "main", "target_sparse_mean")
    main_contrast = median_metric(main_rows, "main", "target_edge_contrast")
    main_empty_fraction = aggregate(main_rows, "main", "empty_exceedance_fraction")

    scan_rows: List[Dict[str, object]] = []
    scan_detail_rows: List[Dict[str, object]] = []
    blob_candidates = {}
    for gamma in (0.05, 0.10, 0.20, 0.30, 0.40):
        set_blob_gamma(blob_model, gamma)
        blob_background, blob_logits, blob_sparse, blob_trace = run_with_trace(blob_model, image_batch)
        blob_empty = run_with_trace(blob_model, empty_batch)[2]
        rows = scene_rows(
            "blob", blob_sparse, blob_trace, mask_array, edge_array, blob_empty, reference_threshold
        )
        for row in rows:
            scan_detail_rows.append({"blob_gamma": gamma, **row})
        ratio = median_metric(rows, "blob", "target_edge_ratio")
        target_response = median_metric(rows, "blob", "target_sparse_mean")
        edge_response = median_metric(rows, "blob", "edge_sparse_mean")
        contrast = median_metric(rows, "blob", "target_edge_contrast")
        empty_fraction = aggregate(rows, "blob", "empty_exceedance_fraction")
        ratio_gate = max(5.0, 1.05 * main_ratio)
        contrast_gain = contrast - main_contrast
        empty_increase = empty_fraction - main_empty_fraction
        scene_pass_fraction = float(
            np.mean(
                [
                    float(row["target_edge_ratio"]) >= ratio_gate
                    and float(row["target_sparse_mean"]) >= 5.0e-4
                    for row in rows
                ]
            )
        )
        passes = (
            ratio >= ratio_gate
            and target_response >= 5.0e-4
            and contrast_gain >= 5.0e-4
            and scene_pass_fraction >= 0.75
            and empty_increase <= 0.001
        )
        scan_rows.append(
            {
                "blob_gamma": gamma,
                "median_target_sparse": target_response,
                "median_edge_sparse": edge_response,
                "median_target_edge_ratio": ratio,
                "ratio_gate": ratio_gate,
                "median_target_edge_contrast": contrast,
                "median_contrast_gain_over_main": contrast_gain,
                "scene_pass_fraction": scene_pass_fraction,
                "mean_empty_exceedance_fraction": empty_fraction,
                "empty_fraction_increase": empty_increase,
                "passes_local_branch_gate": passes,
            }
        )
        blob_candidates[gamma] = (
            rows,
            blob_background,
            blob_logits,
            blob_sparse,
            blob_trace,
        )
    passing = [row for row in scan_rows if row["passes_local_branch_gate"]]
    false_alarm_safe = [
        row for row in scan_rows if float(row["empty_fraction_increase"]) <= 0.001
    ]
    choice_pool = passing if passing else (false_alarm_safe if false_alarm_safe else scan_rows)
    chosen_scan = max(choice_pool, key=lambda row: float(row["median_target_edge_contrast"]))
    chosen_gamma = float(chosen_scan["blob_gamma"])
    chosen_blob = blob_candidates[chosen_gamma]
    blob_rows, blob_background, blob_logits, blob_sparse, blob_trace = chosen_blob
    metric_rows = product_rows + main_rows + blob_rows
    write_csv(results_dir / "mechanism_metrics.csv", metric_rows)
    write_csv(results_dir / "blob_gamma_scan.csv", scan_rows)
    write_csv(results_dir / "blob_gamma_scene_metrics.csv", scan_detail_rows)

    trace_rows: List[Dict[str, object]] = []
    for stage_index, state in enumerate(main_trace, start=1):
        trace_rows.append(
            {
                "stage": stage_index,
                "primal_norm_mean": float(state["primal_norm"].mean()),
                "dual_norm_mean": float(state["dual_norm"].mean()),
                "mu_mean": float(state["mu"].mean()),
                "mu_next_mean": float(state["mu_next"].mean()),
                "alpha_mean": float(state["alpha"].mean()),
                "weight_min": float(state["w_rcp"].min()),
                "weight_max": float(state["w_rcp"].max()),
                "uncertainty_mean": float(state["uncertainty"].mean()),
                "background_correction_ratio": float(
                    state["background_delta"].abs().mean()
                    / (state["background_scale"].mean() + 1.0e-6)
                ),
                "sparse_correction_ratio": float(
                    state["sparse_delta"].abs().mean()
                    / (state["sparse_scale"].mean() + 1.0e-6)
                ),
            }
        )
    write_csv(results_dir / "stage_trace.csv", trace_rows)

    first_mask_indices = np.argwhere(first_target)
    target_row = int(np.round(first_mask_indices[:, 0].mean()))
    traces = {"main": main_trace, "product": product_trace, "blob": blob_trace}
    export_process_figures(figures_dir, main_trace, traces, target_row)

    real_path = image_paths[0]
    real_pil = Image.open(real_path).convert("L").resize((64, 64), _RESAMPLING.BILINEAR)
    real_image = torch.from_numpy(np.asarray(real_pil, dtype=np.float32) / 255.0)[None, None]
    real_output = run_with_trace(main_model, real_image)
    main_first_output = (
        main_background[0:1],
        main_logits[0:1],
        main_sparse[0:1],
        [{key: value[0:1] if value.ndim > 0 and value.shape[0] == args.mechanism_scenes else value for key, value in state.items()} for state in main_trace],
    )

    product_edge = aggregate(product_rows, "product", "edge_sparse_mean")
    main_edge = aggregate(main_rows, "main", "edge_sparse_mean")
    main_edge_gate_value = 1.05 * product_edge
    blob_ratio_gate_value = max(5.0, 1.05 * main_ratio)
    export_result_figures(
        figures_dir,
        first_scene,
        first_target,
        main_first_output,
        metric_rows,
        real_image,
        real_output,
        main_edge_gate_value,
        blob_ratio_gate_value,
        scan_rows,
    )

    elapsed = time.perf_counter() - start
    variants = [
        ("rirufold_rcp", main_model),
        ("rirufold_rcp_product", product_model),
        ("rirufold_rcp_blob", blob_model),
    ]
    complexity_rows = [
        {
            "variant": name,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "stage_num": args.stage_num,
            "hidden_channels": args.hidden_channels,
        }
        for name, model in variants
    ]
    write_csv(results_dir / "model_complexity.csv", complexity_rows)

    summary = {
        "scope": "untrained deterministic mechanism gate; not benchmark accuracy",
        "seed": args.seed,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "nudt_test_images": len(profile_rows),
        "nudt_profiled_intensity_images": min(args.profile_images, len(profile_rows)),
        "nudt_target_components": int(component_areas.size),
        "mechanism_scenes": args.mechanism_scenes,
        "stage_num": args.stage_num,
        "hidden_channels": args.hidden_channels,
        "reference_empty_sparse_quantile_995": reference_threshold,
        "main_edge_leakage_mean": main_edge,
        "product_edge_leakage_mean": product_edge,
        "main_edge_gate_limit": main_edge_gate_value,
        "main_edge_gate_pass": bool(main_edge <= main_edge_gate_value),
        "main_target_edge_ratio_median": main_ratio,
        "main_target_sparse_median": main_target,
        "main_target_edge_contrast_median": main_contrast,
        "exploratory_blob_gamma": chosen_gamma,
        "exploratory_blob_target_sparse_median": median_metric(blob_rows, "blob", "target_sparse_mean"),
        "exploratory_blob_target_edge_ratio_median": median_metric(blob_rows, "blob", "target_edge_ratio"),
        "exploratory_blob_target_edge_contrast_median": median_metric(blob_rows, "blob", "target_edge_contrast"),
        "blob_ratio_gate_limit": blob_ratio_gate_value,
        "blob_scene_pass_fraction": float(chosen_scan["scene_pass_fraction"]),
        "chosen_blob_empty_fraction_increase": float(chosen_scan["empty_fraction_increase"]),
        "blob_local_gate_pass": bool(chosen_scan["passes_local_branch_gate"]),
        "real_trace_image": str(real_path.relative_to(PROJECT_ROOT)),
        "real_trace_resize": [64, 64],
        "mechanism_runtime_seconds": elapsed,
    }
    (results_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
