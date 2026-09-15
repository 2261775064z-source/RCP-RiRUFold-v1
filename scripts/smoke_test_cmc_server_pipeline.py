"""End-to-end CPU smoke test for CMC evaluation and diagnosis CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from RiRUFold_ADMM.models.rirufold_cmc import build_cmc_model  # noqa: E402


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    image = PROJECT_ROOT / "datasets" / "NUDT-SIRST" / "test" / "images" / "000001.png"
    if not image.is_file():
        raise FileNotFoundError(image)
    with tempfile.TemporaryDirectory(prefix="rirufold_cmc_cli_") as raw_temp:
        temp = Path(raw_temp)
        checkpoint = temp / "seed_31" / "checkpoints" / "untrained_stage1.pkl"
        metric = temp / "metrics" / "seed_31" / "nudt" / "rirufold_cmc.json"
        diagnosis = temp / "diagnosis"
        checkpoint.parent.mkdir(parents=True)
        torch.manual_seed(31)
        torch.save(
            build_cmc_model("rirufold_cmc", stage_num=1, hidden_channels=4).state_dict(),
            checkpoint,
        )
        run(
            "scripts/evaluate_rirufold_rcp.py",
            "--net-name", "rirufold_cmc",
            "--checkpoint", str(checkpoint),
            "--dataset", "nudt",
            "--seed", "31",
            "--data-root", str(PROJECT_ROOT / "datasets"),
            "--base-size", "32",
            "--batch-size", "1",
            "--stage-num", "1",
            "--hidden-channels", "4",
            "--max-images", "1",
            "--cpu",
            "--output", str(metric),
        )
        record = json.loads(metric.read_text(encoding="utf-8"))
        assert record["net_name"] == "rirufold_cmc"
        assert record["partial_evaluation"] is True and record["test_images"] == 1
        assert len(record["checkpoint_sha256"]) == 64
        run(
            "scripts/diagnose_rirufold_cmc.py",
            "--net-name", "rirufold_cmc",
            "--checkpoint", str(checkpoint),
            "--image", str(image),
            "--base-size", "32",
            "--stage-num", "1",
            "--hidden-channels", "4",
            "--cpu",
            "--out-dir", str(diagnosis),
        )
        assert (diagnosis / "stage_summary.csv").is_file()
        assert (diagnosis / "stage_01_candidate.png").is_file()
        assert (diagnosis / "stage_01_completion_02.png").is_file()
    print("CMC server evaluation/diagnosis pipeline smoke test passed.")


if __name__ == "__main__":
    main()
