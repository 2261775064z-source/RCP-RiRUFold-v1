"""End-to-end CPU smoke test for evaluation, diagnosis, and aggregation CLIs."""

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

from RiRUFold_ADMM.models.rirufold_rcp import build_rcp_model  # noqa: E402
from RiRUFold_ADMM.models.rirufold_admm import RiRFoldADMM  # noqa: E402


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=PROJECT_ROOT, check=True)


def seed_conflict_is_rejected(checkpoint: Path) -> bool:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rirufold_rcp.py",
            "--net-name", "rirufold_rcp",
            "--checkpoint", str(checkpoint),
            "--dataset", "nudt",
            "--seed", "23",
            "--data-root", str(PROJECT_ROOT / "datasets"),
            "--stage-num", "1",
            "--hidden-channels", "4",
            "--max-images", "1",
            "--cpu",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    return completed.returncode != 0 and "conflicts with --seed 23" in completed.stderr


def main() -> None:
    image = PROJECT_ROOT / "datasets" / "NUDT-SIRST" / "test" / "images" / "000001.png"
    if not image.is_file():
        raise FileNotFoundError(image)

    with tempfile.TemporaryDirectory(prefix="rirufold_rcp_cli_") as raw_temp:
        temp = Path(raw_temp)
        checkpoint = temp / "seed_17" / "checkpoints" / "untrained_stage1.pkl"
        baseline_checkpoint = temp / "seed_17" / "checkpoints" / "untrained_baseline_stage1.pkl"
        metric_17 = temp / "metrics" / "seed_17" / "nudt" / "rirufold_rcp.json"
        baseline_metric = temp / "metrics" / "seed_17" / "nudt" / "rirufold_admm.json"
        diagnosis = temp / "diagnosis"
        checkpoint.parent.mkdir(parents=True)
        torch.manual_seed(17)
        torch.save(
            build_rcp_model("rirufold_rcp", stage_num=1, hidden_channels=4).state_dict(),
            checkpoint,
        )
        torch.save(
            RiRFoldADMM(stage_num=1, hidden_channels=4, use_svt=True).state_dict(),
            baseline_checkpoint,
        )

        run(
            "scripts/evaluate_rirufold_rcp.py",
            "--net-name", "rirufold_rcp",
            "--checkpoint", str(checkpoint),
            "--dataset", "nudt",
            "--seed", "17",
            "--data-root", str(PROJECT_ROOT / "datasets"),
            "--base-size", "32",
            "--batch-size", "1",
            "--stage-num", "1",
            "--hidden-channels", "4",
            "--max-images", "1",
            "--cpu",
            "--output", str(metric_17),
        )
        result = json.loads(metric_17.read_text(encoding="utf-8"))
        assert result["partial_evaluation"] is True
        assert result["test_images"] == 1 and result["full_test_images"] == 664
        assert result["seed"] == 17
        assert len(result["checkpoint_sha256"]) == 64
        assert len(result["test_split_sha256"]) == 64
        assert seed_conflict_is_rejected(checkpoint)

        run(
            "scripts/evaluate_rirufold_rcp.py",
            "--net-name", "rirufold_admm",
            "--checkpoint", str(baseline_checkpoint),
            "--dataset", "nudt",
            "--seed", "17",
            "--data-root", str(PROJECT_ROOT / "datasets"),
            "--base-size", "32",
            "--batch-size", "1",
            "--stage-num", "1",
            "--hidden-channels", "4",
            "--max-images", "1",
            "--cpu",
            "--output", str(baseline_metric),
        )
        baseline_result = json.loads(baseline_metric.read_text(encoding="utf-8"))
        assert baseline_result["net_name"] == "rirufold_admm"
        assert baseline_result["test_images"] == 1
        assert baseline_result["test_split_sha256"] == result["test_split_sha256"]

        run(
            "scripts/diagnose_rirufold_rcp.py",
            "--net-name", "rirufold_rcp",
            "--checkpoint", str(checkpoint),
            "--image", str(image),
            "--base-size", "32",
            "--stage-num", "1",
            "--hidden-channels", "4",
            "--cpu",
            "--out-dir", str(diagnosis),
        )
        assert (diagnosis / "stage_summary.csv").is_file()
        assert (diagnosis / "stage_01_w_rcp.png").is_file()

        second = dict(result)
        second["seed"] = 23
        metric_23 = temp / "metrics" / "seed_23" / "nudt" / "rirufold_rcp.json"
        metric_23.parent.mkdir(parents=True)
        metric_23.write_text(json.dumps(second), encoding="utf-8")
        summary = temp / "summary.csv"
        run(
            "scripts/summarize_server_results.py",
            "--input-root", str(temp / "metrics"),
            "--output", str(summary),
        )
        summary_text = summary.read_text(encoding="utf-8-sig")
        assert "seed_count" in summary_text and "rirufold_rcp" in summary_text

    print("RCP server evaluation/diagnosis/summary pipeline smoke test passed.")


if __name__ == "__main__":
    main()
