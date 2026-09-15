"""Deterministic CMC mechanism study and publication-figure generation.

These experiments use identically initialized, untrained networks to verify
mechanism direction, trace completeness, and falsifiable ablations.  They are
not substitutes for the preregistered three-dataset server comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.ndimage as ndi  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from RiRUFold_ADMM.models.rirufold_cmc import build_cmc_model  # noqa: E402
from utils.plot_style import (  # noqa: E402
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)


VARIANTS = (
    "rirufold_cmc", "rirufold_cmc_nomask", "rirufold_cmc_noside",
    "rirufold_cmc_onepass", "rirufold_cmc_fixedmask",
)
LABELS = {
    "rirufold_cmc": "CMC",
    "rirufold_cmc_nomask": "No mask",
    "rirufold_cmc_noside": "No side",
    "rirufold_cmc_onepass": "One pass",
    "rirufold_cmc_fixedmask": "Fixed mask",
}
COLORS = {
    name: color for name, color in zip(
        VARIANTS,
        (PALETTE["primary"], PALETTE["neutral"], PALETTE["contrast"],
         PALETTE["secondary"], PALETTE["positive"]),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic CMC mechanism study")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--stage-num", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--scene-size", type=int, default=40)
    parser.add_argument("--mechanism-scenes", type=int, default=8)
    parser.add_argument("--profile-images", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty table: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_scene(
    index: int, size: int, target_amplitude: float = 0.42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12091 + index)
    yy, xx = np.mgrid[0:size, 0:size]
    edge_x = size // 2 + (index % 3 - 1) * 3
    ridge_slope = 0.28 + 0.06 * (index % 4)
    ridge = np.exp(-0.5 * ((yy - ridge_slope * xx - size * 0.22) / 1.25) ** 2)
    clean = (
        0.09 + 0.0016 * xx + 0.0010 * yy + 0.13 * (xx >= edge_x)
        + 0.035 * ridge + 0.016 * np.sin(0.17 * xx + 0.21 * yy)
        + rng.normal(0.0, 0.005, size=(size, size))
    )
    center_x = edge_x - 4 + index % 6
    center_y = size // 3 + (index * 4) % max(5, size // 3)
    # NUDT's median target fraction is about 39/65536 ~= 0.00060.  At
    # 40x40 this corresponds to roughly one pixel, so the scoring support must
    # not be the much larger blobs often used in generic synthetic demos.
    radius = 0.70 + 0.08 * (index % 3)
    blob = np.exp(-0.5 * (((xx - center_x) / radius) ** 2 + ((yy - center_y) / radius) ** 2))
    target = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius**2
    image = clean + target_amplitude * blob
    edge = ((np.abs(xx - edge_x) <= 2) | (np.abs(yy - ridge_slope * xx - size * 0.22) <= 1.5))
    edge &= ~target
    return (
        np.clip(image, 0.0, 1.0).astype(np.float32),
        np.clip(clean, 0.0, 1.0).astype(np.float32),
        target,
        edge,
    )


def make_models(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    main = build_cmc_model(
        "rirufold_cmc", stage_num=args.stage_num, hidden_channels=args.hidden_channels
    ).eval()
    models = {"rirufold_cmc": main}
    for name in VARIANTS[1:]:
        model = build_cmc_model(
            name, stage_num=args.stage_num, hidden_channels=args.hidden_channels
        ).eval()
        model.load_state_dict(main.state_dict())
        models[name] = model
    return models


def run_trace(model: torch.nn.Module, image: torch.Tensor):
    with torch.no_grad():
        background, logits, trace = model(image, return_trace=True)
        _, _, aux = model(image, return_aux=True)
    return background, logits, aux["sparse_intensity"].detach(), trace


def collect_nudt_contrast(limit: int):
    image_dir = PROJECT_ROOT / "datasets" / "NUDT-SIRST" / "test" / "images"
    mask_dir = PROJECT_ROOT / "datasets" / "NUDT-SIRST" / "test" / "masks"
    image_paths = sorted(image_dir.glob("*.png"))[:limit]
    if not image_paths:
        raise RuntimeError("local NUDT-SIRST test images are missing")
    target_values, background_values, rows = [], [], []
    for path in image_paths:
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_dir / path.name).convert("L"), dtype=np.uint8) > 0
        contrast = np.maximum(image - ndi.uniform_filter(image, size=9, mode="reflect"), 0.0)
        if mask.any():
            target_values.append(contrast[mask])
        background_values.append(contrast[~mask][::128])
        rows.append(
            {
                "file": path.name,
                "image_sha256": sha256_file(path),
                "mask_sha256": sha256_file(mask_dir / path.name),
                "target_pixels": int(mask.sum()),
                "target_contrast_mean": float(contrast[mask].mean()) if mask.any() else 0.0,
                "background_contrast_mean": float(contrast[~mask].mean()),
            }
        )
    return np.concatenate(target_values), np.concatenate(background_values), rows


def mechanism_rows(
    models: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    clean: np.ndarray,
    targets: np.ndarray,
    edges: np.ndarray,
):
    all_rows, outputs = [], {}
    for name, model in models.items():
        background, logits, sparse, trace = run_trace(model, images)
        outputs[name] = (background, logits, sparse, trace)
        b_np = background[:, 0].cpu().numpy()
        s_np = sparse[:, 0].cpu().numpy()
        for index in range(len(clean)):
            target = targets[index]
            non_target = ~target
            all_rows.append(
                {
                    "scene": index,
                    "variant": name,
                    "target_absorption_mae": float(np.abs(b_np[index] - clean[index])[target].mean()),
                    "background_mae": float(np.abs(b_np[index] - clean[index])[non_target].mean()),
                    "target_sparse_mean": float(s_np[index][target].mean()),
                    "edge_sparse_mean": float(s_np[index][edges[index]].mean()),
                    "target_edge_contrast": float(
                        s_np[index][target].mean() - s_np[index][edges[index]].mean()
                    ),
                    "candidate_target_mean": float(trace[-1]["candidate"][index, 0].cpu().numpy()[target].mean()),
                    "candidate_background_mean": float(trace[-1]["candidate"][index, 0].cpu().numpy()[non_target].mean()),
                    "decomposition_mae": float(
                        np.abs(images[index, 0].cpu().numpy() - b_np[index] - s_np[index]).mean()
                    ),
                    "counterfactual_loss": float(trace[-1]["counterfactual_loss"]),
                }
            )
    return all_rows, outputs


def collect_stage_rows(trace, target: np.ndarray, clean: np.ndarray):
    rows = []
    for index, state in enumerate(trace, start=1):
        background = state["background"][0, 0].cpu().numpy()
        rows.append(
            {
                "stage": index,
                "candidate_mean": float(state["candidate"].mean()),
                "candidate_target_mean": float(state["candidate"][0, 0].cpu().numpy()[target].mean()),
                "candidate_schedule": float(state["candidate_fraction"]),
                "target_absorption_mae": float(np.abs(background - clean)[target].mean()),
                "primal_norm": float(state["primal_norm"].mean()),
                "dual_norm": float(state["dual_norm"].mean()),
                "mu_next": float(state["mu_next"].mean()),
                "p_x": float(state["precision"][0]),
                "p_l": float(state["precision"][1]),
                "p_a": float(state["precision"][2]),
                "counterfactual_loss": float(state["counterfactual_loss"]),
                "budget_loss": float(state["budget_loss"]),
            }
        )
    return rows


def collect_mask_scan(model: torch.nn.Module, size: int):
    rows = []
    for amplitude in (0.15, 0.25, 0.35, 0.45, 0.55):
        image, _, target, _ = synthetic_scene(41, size, target_amplitude=amplitude)
        tensor = torch.from_numpy(image)[None, None]
        _, _, _, trace = run_trace(model, tensor)
        candidate = trace[0]["candidate"][0, 0].cpu().numpy()
        rows.append(
            {
                "target_amplitude": amplitude,
                "target_candidate_mean": float(candidate[target].mean()),
                "background_candidate_mean": float(candidate[~target].mean()),
                "candidate_mean": float(candidate.mean()),
                "scheduled_fraction": float(trace[0]["candidate_fraction"]),
            }
        )
    return rows


def collect_complexity(models):
    """Record deterministic architecture complexity; runtime belongs to server eval."""
    rows = []
    for name, model in models.items():
        first_stage = model.stages[0]
        rows.append(
            {
                "variant": name,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "stage_num": model.stage_num,
                "inner_steps": first_stage.inner_steps,
                "mask_enabled": int(first_stage.use_mask),
                "side_prior_enabled": int(first_stage.use_side_prior),
                "fixed_mask": int(model.fixed_mask),
            }
        )
    return rows


def hide(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def export_figures(
    figures_dir: Path,
    scene,
    clean,
    target,
    edge,
    outputs,
    metric_rows,
    stage_rows,
    target_contrast,
    background_contrast,
):
    main_background, main_logits, main_sparse, main_trace = outputs["rirufold_cmc"]
    row = int(np.argwhere(target)[0, 0])
    x = np.arange(scene.shape[1])

    fig, axes = publication_subplots(1, 4, width="double", aspect=0.28)
    for axis, value, title, cmap in zip(
        axes, (scene, clean, target, edge),
        ("Input", "Clean background", "Target support", "Clutter support"),
        ("gray", "gray", "gray", "gray"),
    ):
        axis.imshow(value, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        hide(axis)
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "raw_q2_synthetic_challenges")
    plt.close(fig)

    fig, axis = publication_subplots(width="report", aspect=0.52)
    state0 = main_trace[0]
    axis.plot(x, state0["candidate_score"][0, 0, row].cpu(), color=PALETTE["neutral"], label="Candidate score")
    axis.plot(x, state0["candidate"][0, 0, row].cpu(), color=PALETTE["contrast"], linestyle="-.", label="C: excluded")
    axis.plot(x, state0["valid"][0, 0, row].cpu(), color=PALETTE["primary"], linestyle="--", label="V: observed")
    axis.set(title="Candidate profile", xlabel="Horizontal pixel", ylabel="Normalized value")
    axis.legend(ncols=3, loc="upper center")
    export_figure(fig, figures_dir / "raw_q2_candidate_score_profiles")
    plt.close(fig)

    fig, axis = publication_subplots(width="report", aspect=0.52)
    for values, label, color, line_style in (
        (target_contrast, "Target pixels", PALETTE["contrast"], "-"),
        (background_contrast, "Background sample", PALETTE["primary"], "--"),
    ):
        sorted_values = np.sort(values)
        axis.plot(
            sorted_values, np.arange(1, len(values) + 1) / len(values),
            color=color, linestyle=line_style, label=label,
        )
    axis.set(title="NUDT local-contrast ECDF", xlabel="Positive 9x9 local contrast", ylabel="Cumulative fraction")
    axis.legend()
    export_figure(fig, figures_dir / "raw_q2_nudt_local_contrast")
    plt.close(fig)

    sequence = (
        (state0["candidate"][0, 0].cpu(), "Candidate C"),
        (state0["valid"][0, 0].cpu(), "Observed V"),
        (state0["side_prior"][0, 0].cpu(), "Side prior"),
        (state0["completion_inner"][0, 1, 0].cpu(), "Completion 1"),
        (state0["completion_inner"][0, 2, 0].cpu(), "Completion 2"),
    )
    fig, axes = publication_subplots(1, 5, width="double", aspect=0.24)
    for axis, (value, title) in zip(axes, sequence):
        axis.imshow(value, cmap="gray")
        axis.set_title(title)
        hide(axis)
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "process_q2_mask_completion_sequence")
    plt.close(fig)

    stages = np.asarray([row_["stage"] for row_ in stage_rows])
    fig, axes = publication_subplots(1, 2, width="report", aspect=0.48)
    for field, label, color, marker in (
        ("p_x", "Input", PALETTE["primary"], "o"),
        ("p_l", "Low-rank", PALETTE["positive"], "s"),
        ("p_a", "Side prior", PALETTE["contrast"], "^"),
    ):
        axes[0].plot(stages, [r[field] for r in stage_rows], marker=marker, color=color, label=label)
    axes[0].set(title="Positive precisions", xlabel="Stage", ylabel="Precision", xticks=stages)
    axes[0].legend(ncols=3)
    axes[1].plot(stages, [r["candidate_mean"] for r in stage_rows], marker="o", color=PALETTE["contrast"], label="Mask mean")
    axes[1].plot(stages, [r["candidate_schedule"] for r in stage_rows], marker="s", color=PALETTE["neutral"], linestyle="--", label="Area schedule")
    axes[1].set(title="Candidate budget", xlabel="Stage", ylabel="Spatial fraction", xticks=stages)
    axes[1].legend()
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "process_q2_precision_components")
    plt.close(fig)

    fig, axes = publication_subplots(1, 2, width="report", aspect=0.48)
    primal = np.asarray([r["primal_norm"] for r in stage_rows])
    dual = np.asarray([r["dual_norm"] for r in stage_rows])
    axes[0].plot(stages, primal / max(primal[0], 1e-8), marker="o", color=PALETTE["primary"], label="Primal")
    axes[0].plot(stages, dual / max(dual[0], 1e-8), marker="s", color=PALETTE["contrast"], label="Dual proxy")
    axes[0].set(title="Residual trajectory", xlabel="Stage", ylabel="Value / stage 1", xticks=stages)
    axes[0].legend()
    axes[1].plot(stages, [r["counterfactual_loss"] for r in stage_rows], marker="o", color=PALETTE["positive"], label="Counterfactual")
    axes[1].plot(stages, [r["budget_loss"] for r in stage_rows], marker="s", color=PALETTE["secondary"], label="Mask budget")
    axes[1].set(title="Auxiliary objectives", xlabel="Stage", ylabel="Untrained loss", xticks=stages)
    axes[1].legend()
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "process_q2_stage_residuals")
    plt.close(fig)

    probability = torch.sigmoid(main_logits[0, 0]).cpu()
    result_panels = (
        (scene, "Input"), (clean, "Clean reference"),
        (main_background[0, 0].cpu(), "CMC background"),
        (main_sparse[0, 0].cpu(), "Physical sparse"), (probability, "Probability"),
    )
    fig, axes = publication_subplots(1, 5, width="double", aspect=0.24)
    for axis, (value, title) in zip(axes, result_panels):
        axis.imshow(value, cmap="gray")
        axis.set_title(title)
        hide(axis)
    add_panel_labels(axes)
    fig.suptitle("Untrained mechanism output", fontsize=8)
    export_figure(fig, figures_dir / "result_q2_counterfactual_decomposition")
    plt.close(fig)

    fig, axes = publication_subplots(1, 2, width="report", aspect=0.50)
    rng = np.random.default_rng(7)
    for position, name in enumerate(VARIANTS):
        selected = [r for r in metric_rows if r["variant"] == name]
        for axis, field in zip(axes, ("target_absorption_mae", "target_edge_contrast")):
            values = np.asarray([r[field] for r in selected])
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axis.scatter(position + jitter, values, s=12, color=COLORS[name], alpha=0.75)
            axis.plot([position - 0.18, position + 0.18], [np.median(values)] * 2, color=PALETTE["dark"], linewidth=1.2)
    ticks = np.arange(len(VARIANTS))
    tick_labels = [LABELS[name] for name in VARIANTS]
    axes[0].set(title="Target absorption", ylabel="Background MAE on target", xticks=ticks, xticklabels=tick_labels)
    axes[1].set(title="Target-clutter separation", ylabel="Sparse contrast", xticks=ticks, xticklabels=tick_labels)
    for axis in axes:
        axis.tick_params(axis="x", rotation=22)
    add_panel_labels(axes)
    export_figure(fig, figures_dir / "result_q2_mechanism_metrics")
    plt.close(fig)

    fig, axis = publication_subplots(width="report", aspect=0.54)
    label_offsets = {
        "rirufold_cmc": (-8, -16, "right"),
        "rirufold_cmc_nomask": (-8, 10, "right"),
        "rirufold_cmc_noside": (8, -8, "left"),
        "rirufold_cmc_onepass": (25, 0, "left"),
        "rirufold_cmc_fixedmask": (20, -18, "left"),
    }
    for name in VARIANTS:
        selected = [r for r in metric_rows if r["variant"] == name]
        absorption = np.median([r["target_absorption_mae"] for r in selected])
        separation = np.median([r["target_edge_contrast"] for r in selected])
        axis.scatter(absorption, separation, s=32, color=COLORS[name])
        dx, dy, horizontal = label_offsets[name]
        axis.annotate(
            LABELS[name], (absorption, separation), xytext=(dx, dy),
            textcoords="offset points", fontsize=7, ha=horizontal,
        )
    axis.set(title="Ablation mechanism trade-off", xlabel="Target absorption MAE (lower)", ylabel="Target-edge sparse contrast (higher)")
    export_figure(fig, figures_dir / "result_q2_ablation_tradeoff")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.mechanism_scenes < 3 or args.scene_size < 24:
        raise ValueError("use at least 3 scenes and a scene size of at least 24")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    apply_publication_style(language="en", width="report")
    results_dir = PROJECT_ROOT / "results"
    figures_dir = PROJECT_ROOT / "figures"

    generated = [synthetic_scene(i, args.scene_size) for i in range(args.mechanism_scenes)]
    scenes, clean, targets, edges = map(np.stack, zip(*generated))
    images = torch.from_numpy(scenes[:, None])
    models = make_models(args)
    metric_rows, outputs = mechanism_rows(models, images, clean, targets, edges)
    main_trace = outputs["rirufold_cmc"][3]
    stage_rows = collect_stage_rows(main_trace, targets[0], clean[0])
    mask_rows = collect_mask_scan(models["rirufold_cmc"], args.scene_size)
    complexity_rows = collect_complexity(models)
    target_contrast, background_contrast, profile_rows = collect_nudt_contrast(args.profile_images)

    write_csv(results_dir / "cmc_mechanism_metrics.csv", metric_rows)
    write_csv(results_dir / "cmc_stage_trace.csv", stage_rows)
    write_csv(results_dir / "cmc_mask_scan.csv", mask_rows)
    write_csv(results_dir / "cmc_model_complexity.csv", complexity_rows)
    write_csv(results_dir / "cmc_nudt_profile.csv", profile_rows)
    export_figures(
        figures_dir, scenes[0], clean[0], targets[0], edges[0], outputs,
        metric_rows, stage_rows, target_contrast, background_contrast,
    )

    medians = {}
    for name in VARIANTS:
        selected = [row for row in metric_rows if row["variant"] == name]
        medians[name] = {
            key: float(np.median([float(row[key]) for row in selected]))
            for key in ("target_absorption_mae", "background_mae", "target_sparse_mean", "edge_sparse_mean", "target_edge_contrast")
        }
    summary = {
        "scope": "deterministic untrained mechanism validation; not benchmark accuracy",
        "seed": args.seed,
        "stage_num": args.stage_num,
        "hidden_channels": args.hidden_channels,
        "scene_size": args.scene_size,
        "mechanism_scenes": args.mechanism_scenes,
        "nudt_profile_images": args.profile_images,
        "variant_medians": medians,
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": "cpu",
        },
        "source_sha256": {
            "model": sha256_file(PROJECT_ROOT / "models" / "rirufold_cmc.py"),
            "script": sha256_file(Path(__file__)),
            "analysis": sha256_file(PROJECT_ROOT / "题目分析报告.md"),
        },
        "reproduce": ".venv/Scripts/python.exe scripts/run_cmc_research.py --seed {}".format(args.seed),
    }
    (results_dir / "cmc_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
