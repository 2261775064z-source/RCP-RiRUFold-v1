"""Evaluate a trained ADMM/RCP/CMC checkpoint on an explicit test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.utils.data as data
from PIL import Image


_RESAMPLING = getattr(Image, "Resampling", Image)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from RiRUFold_ADMM.models.rirufold_rcp import build_rcp_model  # noqa: E402
from RiRUFold_ADMM.models.rirufold_cmc import build_cmc_model  # noqa: E402
from RiRUFold_ADMM.models.rirufold_admm import RiRFoldADMM  # noqa: E402


def _resolve_pair_layout(base_dir: Path):
    """Resolve a test split without inventing or mixing train/test membership."""
    candidates = [base_dir, base_dir / base_dir.name]
    for candidate in candidates:
        image_dir = candidate / "test" / "images"
        mask_dir = candidate / "test" / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            names = sorted(path.name for path in image_dir.iterdir() if path.is_file())
            return image_dir, mask_dir, names

        image_dir = candidate / "images"
        mask_dir = candidate / "masks"
        if not (image_dir.is_dir() and mask_dir.is_dir()):
            image_dir = candidate / "IRSTD1k_Img"
            mask_dir = candidate / "IRSTD1k_Label"
        split_file = candidate / "test.txt"
        if image_dir.is_dir() and mask_dir.is_dir() and split_file.is_file():
            names = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            return image_dir, mask_dir, names
    raise FileNotFoundError("No explicit test split found under {}".format(base_dir))


def _resolve_named_file(directory: Path, name: str) -> Path:
    direct = directory / name
    candidates = [direct] if direct.suffix else [directory / (name + suffix) for suffix in (".png", ".jpg", ".bmp")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Cannot resolve {} under {}".format(name, directory))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _split_sha256(dataset: "TestSplitDataset") -> str:
    """Hash ordered membership plus image/mask bytes for the explicit test split."""
    digest = hashlib.sha256()
    for name in dataset.names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        for directory in (dataset.image_dir, dataset.mask_dir):
            path = _resolve_named_file(directory, name)
            digest.update(path.name.encode("utf-8"))
            digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest().upper()


def _validate_seed_path(checkpoint: Path, seed: int) -> None:
    for part in checkpoint.resolve().parts:
        if part.startswith("seed_") and part[5:].isdigit():
            path_seed = int(part[5:])
            if path_seed != seed:
                raise ValueError(
                    "checkpoint path seed_{} conflicts with --seed {}".format(path_seed, seed)
                )


class TestSplitDataset(data.Dataset):
    def __init__(self, base_dir: Path, base_size: int) -> None:
        self.image_dir, self.mask_dir, self.names = _resolve_pair_layout(base_dir)
        self.base_size = int(base_size)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = Image.open(_resolve_named_file(self.image_dir, name)).convert("L").resize(
            (self.base_size, self.base_size), _RESAMPLING.BILINEAR
        )
        mask = Image.open(_resolve_named_file(self.mask_dir, name)).convert("L").resize(
            (self.base_size, self.base_size), _RESAMPLING.NEAREST
        )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(image_array[None]), torch.from_numpy(mask_array[None])


class DetectionMeter:
    """Micro pixel scores plus 8-connected target Pd and background-pixel Fa."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.true_targets = self.target_count = 0
        self.false_pixels = self.background_pixels = 0
        self.structure = np.ones((3, 3), dtype=np.uint8)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = prediction.astype(bool)
        target = target.astype(bool)
        self.tp += int(np.logical_and(prediction, target).sum())
        self.fp += int(np.logical_and(prediction, ~target).sum())
        self.fn += int(np.logical_and(~prediction, target).sum())
        labeled, count = ndi.label(target, structure=self.structure)
        self.target_count += int(count)
        self.background_pixels += int((labeled == 0).sum())
        self.false_pixels += int(np.logical_and(prediction, labeled == 0).sum())
        for component in range(1, count + 1):
            self.true_targets += int(np.logical_and(prediction, labeled == component).any())

    def get(self):
        eps = np.finfo(np.float64).eps
        miou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        pd_value = self.true_targets / (self.target_count + eps)
        fa_value = self.false_pixels / (self.background_pixels + eps)
        return miou, precision, recall, f1, pd_value, fa_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent RiRUFold checkpoint evaluation")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--net-name",
        default="rirufold_rcp",
        choices=[
            "rirufold_admm",
            "rirufold_rcp",
            "rirufold_rcp_product",
            "rirufold_rcp_fixedgate",
            "rirufold_rcp_mean",
            "rirufold_rcp_nofeedback",
            "rirufold_rcp_blob",
            "rirufold_rcp_global",
            "rirufold_rcp_signed",
            "rirufold_cmc",
            "rirufold_cmc_nomask",
            "rirufold_cmc_noside",
            "rirufold_cmc_onepass",
            "rirufold_cmc_fixedmask",
        ],
    )
    parser.add_argument("--dataset", required=True, choices=["nudt", "irstd1k", "sirstaug"])
    parser.add_argument("--seed", required=True, type=int, help="training seed stored in the result record")
    parser.add_argument("--data-root", type=Path, default=Path("./datasets"))
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stage-num", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-images",
        type=int,
        help="optional positive subset size for CLI smoke tests; omit for publication evaluation",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must lie inside (0, 1)")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive when provided")
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _validate_seed_path(args.checkpoint, args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset_names = {"nudt": "NUDT-SIRST", "irstd1k": "IRSTD-1k", "sirstaug": "sirst_aug"}
    full_dataset = TestSplitDataset(args.data_root / dataset_names[args.dataset], args.base_size)
    full_test_images = len(full_dataset)
    test_split_sha256 = _split_sha256(full_dataset)
    dataset = full_dataset
    if args.max_images is not None:
        dataset = data.Subset(full_dataset, range(min(args.max_images, full_test_images)))
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.net_name == "rirufold_admm":
        model = RiRFoldADMM(
            stage_num=args.stage_num,
            hidden_channels=args.hidden_channels,
            use_svt=True,
        ).to(device)
    elif args.net_name.startswith("rirufold_rcp"):
        model = build_rcp_model(
            args.net_name,
            stage_num=args.stage_num,
            hidden_channels=args.hidden_channels,
        ).to(device)
    else:
        model = build_cmc_model(
            args.net_name,
            stage_num=args.stage_num,
            hidden_channels=args.hidden_channels,
        ).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device))
    model.eval()

    metric = DetectionMeter()
    elapsed = 0.0
    image_count = 0
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _, target_logits = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            image_count += image.shape[0]
            probability = torch.sigmoid(target_logits).cpu().numpy()
            label_array = label.numpy()
            for probability_item, label_item in zip(probability, label_array):
                metric.update(
                    probability_item.squeeze(0) > args.threshold,
                    label_item.squeeze(0) > 0.5,
                )

    miou, precision, recall, f1, pd_value, fa_value = metric.get()
    result = {
        "scope": (
            "partial checkpoint CLI smoke evaluation"
            if args.max_images is not None
            else "trained checkpoint full test split evaluation"
        ),
        "dataset": args.dataset,
        "seed": args.seed,
        "net_name": args.net_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "test_split_sha256": test_split_sha256,
        "threshold": args.threshold,
        "test_images": image_count,
        "full_test_images": full_test_images,
        "partial_evaluation": args.max_images is not None,
        "mIoU": float(miou),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "Pd": float(pd_value),
        "Fa_1e6_per_pixel": float(fa_value * 1.0e6),
        "seconds_per_image": elapsed / max(image_count, 1),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "device": str(device),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
