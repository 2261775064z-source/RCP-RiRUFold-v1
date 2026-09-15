"""Synthetic complete-matrix smoke test for server result analysis."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("nudt", "irstd1k", "sirstaug")
MODELS = (
    "rirufold_admm", "rirufold_rcp", "rirufold_cmc_nomask",
    "rirufold_cmc_noside", "rirufold_cmc_onepass",
    "rirufold_cmc_fixedmask", "rirufold_cmc",
)
SEEDS = (42, 3407, 2026)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cmc_result_analysis_") as raw_temp:
        root = Path(raw_temp)
        metrics = root / "metrics"
        output = root / "analysis"
        mpl_config = root / "mplconfig"
        mpl_config.mkdir()
        for d_index, dataset in enumerate(DATASETS):
            for m_index, model in enumerate(MODELS):
                for s_index, seed in enumerate(SEEDS):
                    value = 0.55 + 0.01 * d_index + 0.002 * m_index + 0.0001 * s_index
                    record = {
                        "dataset": dataset,
                        "net_name": model,
                        "seed": seed,
                        "threshold": 0.5,
                        "partial_evaluation": False,
                        "test_images": 100,
                        "full_test_images": 100,
                        "checkpoint_sha256": hashlib.sha256(
                            f"{dataset}/{model}/{seed}".encode("utf-8")
                        ).hexdigest().upper(),
                        "test_split_sha256": (str(d_index + 1) * 64)[:64],
                        "mIoU": value,
                        "precision": value + 0.03,
                        "recall": value + 0.02,
                        "F1": value + 0.025,
                        "Pd": value + 0.04,
                        "Fa_1e6_per_pixel": 18.0 - m_index + 0.1 * s_index,
                        "seconds_per_image": 0.01 + 0.001 * m_index,
                        "trainable_parameters": 1000 + 100 * m_index,
                    }
                    path = metrics / f"seed_{seed}" / dataset / f"{model}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(record), encoding="utf-8")
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = str(mpl_config)
        subprocess.run(
            [
                sys.executable, "scripts/analyze_cmc_server_results.py",
                "--input-root", str(metrics), "--output-dir", str(output),
                "--bootstrap-repeats", "1000",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env=environment,
        )
        expected = (
            "server_seed_records.csv", "server_summary.csv", "server_paired_gains.csv",
            "result_q2_server_main.png", "result_q2_server_main.svg",
            "result_q2_server_ablation.png", "result_q2_server_ablation.svg",
            "result_q2_server_paired_gain.png", "result_q2_server_paired_gain.svg",
            "CMC服务器结果报告.md",
        )
        missing = [name for name in expected if not (output / name).is_file()]
        assert not missing, f"missing analysis outputs: {missing}"
        report = (output / "CMC服务器结果报告.md").read_text(encoding="utf-8")
        assert "3 数据集 × 7 模型 × 3 种子" in report
        assert "CMC 相对 RCP" in report
    print("CMC strict server-result analysis smoke test passed.")


if __name__ == "__main__":
    main()
