"""Export candidate, completion, consensus, and ADMM traces from a CMC checkpoint."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_RESAMPLING = getattr(Image, "Resampling", Image)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from RiRUFold_ADMM.models.rirufold_cmc import build_cmc_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained CMC-RiRUFold stage traces")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--net-name",
        default="rirufold_cmc",
        choices=[
            "rirufold_cmc", "rirufold_cmc_nomask", "rirufold_cmc_noside",
            "rirufold_cmc_onepass", "rirufold_cmc_fixedmask",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./analysis_outputs/rirufold_cmc_trace"))
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--stage-num", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def save_gray(path: Path, value: torch.Tensor, signed: bool = False) -> None:
    array = value.detach().float().cpu().squeeze().numpy()
    if signed:
        scale = max(float(np.abs(array).max()), 1.0e-6)
        normalized = 0.5 + 0.5 * array / scale
    else:
        low, high = np.percentile(array, (1, 99))
        normalized = (array - low) / max(float(high - low), 1.0e-6)
    Image.fromarray(np.clip(normalized * 255.0, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    raw = Image.open(args.image).convert("L").resize(
        (args.base_size, args.base_size), _RESAMPLING.BILINEAR
    )
    image = torch.from_numpy(np.asarray(raw, dtype=np.float32) / 255.0)[None, None].to(device)
    model = build_cmc_model(
        args.net_name, stage_num=args.stage_num, hidden_channels=args.hidden_channels
    ).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device))
    model.eval()
    with torch.no_grad():
        background, logits, trace = model(image, return_trace=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_gray(args.out_dir / "input.png", image)
    save_gray(args.out_dir / "final_background.png", background)
    save_gray(args.out_dir / "final_sparse_intensity.png", trace[-1]["sparse_intensity"])
    save_gray(args.out_dir / "final_probability.png", torch.sigmoid(logits))
    rows = []
    for index, state in enumerate(trace, start=1):
        prefix = "stage_{:02d}".format(index)
        for name, signed in (
            ("candidate_score", False), ("candidate", False), ("valid", False),
            ("side_prior", False), ("background_bar", False), ("background", False),
            ("sparse_intensity", False), ("w_cmc", False),
            ("q_background", True), ("q_sparse", True), ("primal_residual", True),
        ):
            save_gray(args.out_dir / "{}_{}.png".format(prefix, name), state[name], signed)
        for inner_index in range(state["completion_inner"].shape[1]):
            save_gray(
                args.out_dir / "{}_completion_{:02d}.png".format(prefix, inner_index),
                state["completion_inner"][:, inner_index],
            )
        rows.append(
            {
                "stage": index,
                "candidate_mean": float(state["candidate"].mean()),
                "candidate_target_fraction": float(state["candidate_fraction"]),
                "primal_norm": float(state["primal_norm"].mean()),
                "dual_norm": float(state["dual_norm"].mean()),
                "mu": float(state["mu"].mean()),
                "mu_next": float(state["mu_next"].mean()),
                "p_x": float(state["precision"][0]),
                "p_l": float(state["precision"][1]),
                "p_a": float(state["precision"][2]),
                "fusion_denominator_min": float(state["precision_denominator"].min()),
                "side_enabled": bool(state["side_enabled"]),
                "counterfactual_loss": float(state["counterfactual_loss"]),
            }
        )
    with (args.out_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved CMC trace to {}".format(args.out_dir.resolve()))


if __name__ == "__main__":
    main()
