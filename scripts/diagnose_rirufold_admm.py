"""Export real stage-wise diagnostics for a trained RiRFoldADMM checkpoint.

The script deliberately does not calculate benchmark metrics. Its role is to
produce evidence for the unfolding interpretation: background/sparse states,
WLSE factors, primal/dual residuals, and penalty/dual-step trajectories.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export RiRFoldADMM stage traces for selected infrared images.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--net-name", default="rirufold_admm", help="Registered RiRFoldADMM variant.")
    parser.add_argument("--image", required=True, type=Path, help="One grayscale test image.")
    parser.add_argument("--out-dir", type=Path, default=Path("./analysis_outputs/rirufold_admm_trace"))
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--stage-num", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def save_gray(path: Path, tensor: torch.Tensor, signed: bool = False) -> None:
    array = tensor.detach().float().cpu().squeeze().numpy()
    if signed:
        scale = max(float(np.abs(array).max()), 1e-6)
        array = 0.5 + 0.5 * array / scale
    else:
        lo, hi = np.percentile(array, (1, 99))
        array = (array - lo) / max(float(hi - lo), 1e-6)
    cv2.imwrite(str(path), np.clip(array * 255.0, 0, 255).astype(np.uint8))


def main():
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    raw = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Cannot read image: {args.image}")
    raw = cv2.resize(raw, (args.base_size, args.base_size), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(raw.astype(np.float32) / 255.0).view(1, 1, args.base_size, args.base_size).to(device)

    net = get_model(
        args.net_name,
        stage_num=args.stage_num,
        hidden_channels=args.hidden_channels,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    net.load_state_dict(state)
    net.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_gray(args.out_dir / "input.png", image)
    with torch.no_grad():
        background, sparse, trace = net(image, return_trace=True)
        probability = torch.sigmoid(sparse)
    save_gray(args.out_dir / "final_background.png", background)
    save_gray(args.out_dir / "final_sparse_logits.png", sparse, signed=True)
    save_gray(args.out_dir / "final_probability.png", probability)

    summary_rows = []
    for index, state in enumerate(trace, start=1):
        prefix = f"stage_{index:02d}"
        for name, signed in (
            ("background", False),
            ("sparse", True),
            ("w_ls", False),
            ("w_se", False),
            ("w_lse", False),
            ("coherence", False),
            ("primal_residual", True),
        ):
            save_gray(args.out_dir / f"{prefix}_{name}.png", state[name], signed=signed)
        summary_rows.append({
            "stage": index,
            "primal_residual": float(state["primal_norm"].cpu()),
            "dual_residual": float(state["dual_norm"].cpu()),
            "mu": float(state["mu"].cpu()),
            "mu_next": float(state["mu_next"].cpu()),
            "alpha_mean": float(state["alpha"].mean().cpu()),
            "w_ls_mean": float(state["w_ls"].mean().cpu()),
            "w_se_mean": float(state["w_se"].mean().cpu()),
            "w_lse_mean": float(state["w_lse"].mean().cpu()),
            "background_gate": float(state["background_gate"].cpu()),
            "sparse_gate": float(state["sparse_gate"].cpu()),
        })

    summary_path = args.out_dir / "stage_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Device: {device}")
    print(f"Saved stage trace: {args.out_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
