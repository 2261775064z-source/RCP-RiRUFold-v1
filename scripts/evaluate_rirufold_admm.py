"""Independent NUDT-SIRST/IRSTD-1K evaluation for RiRFoldADMM checkpoints."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.utils.data as data

from models import get_model
from utils.data import IRSTD1kDataset, NUDTDataset, SirstAugDataset
from utils.metrics import SegmentationMetricTPFNFP
from utils.my_pd_fa import my_PD_FA


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a RiRFoldADMM checkpoint on an independent test split.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--net-name", default="rirufold_admm", help="Registered RiRFoldADMM variant.")
    parser.add_argument("--dataset", required=True, choices=["nudt", "irstd1k", "sirstaug"])
    parser.add_argument("--data-root", type=Path, default=Path("./datasets"))
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stage-num", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.5)
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


def main():
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.dataset == "nudt":
        dataset = NUDTDataset(base_dir=args.data_root / "NUDT-SIRST", mode="test", base_size=args.base_size)
    elif args.dataset == "irstd1k":
        dataset = IRSTD1kDataset(base_dir=args.data_root / "IRSTD-1k", mode="test", base_size=args.base_size)
    else:
        dataset = SirstAugDataset(base_dir=args.data_root / "sirst_aug", mode="test", base_size=args.base_size)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    net = get_model(
        args.net_name,
        stage_num=args.stage_num,
        hidden_channels=args.hidden_channels,
    ).to(device)
    net.load_state_dict(load_checkpoint(args.checkpoint, device))
    net.eval()

    metric = SegmentationMetricTPFNFP(nclass=1)
    pd_fa = my_PD_FA()
    logit_threshold = math.log(args.threshold / (1.0 - args.threshold))
    with torch.no_grad():
        for image, label in loader:
            _, target_logits = net(image.to(device))
            probability = torch.sigmoid(target_logits).cpu()
            metric.update(label, target_logits.cpu() - logit_threshold)
            for prob, gt in zip(probability, label):
                # Remove the singleton channel before connected-component PD/FA.
                pd_fa.update((prob.squeeze(0).numpy() > args.threshold), gt.squeeze(0).numpy())

    miou, precision, recall, f1 = metric.get()
    pd, fa = pd_fa.get()
    print(
        f"Result dataset={args.dataset}: mIoU={miou:.6f}, "
        f"precision={precision:.6f}, recall={recall:.6f}, F1={f1:.6f}, "
        f"Pd={pd:.6f}, Fa={fa * 1e6:.6f} (1e-6/pixel)"
    )


if __name__ == "__main__":
    main()
