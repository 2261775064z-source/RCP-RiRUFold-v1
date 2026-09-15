"""Single local command for both RCP and CMC mechanism packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = (
    "README.md",
    "figure_contracts.md",
    "models/__init__.py",
    "models/rirufold_rcp.py",
    "models/rirufold_cmc.py",
    "utils/plot_style.py",
    "scripts/reproduce_all.py",
    "scripts/run_rcp_research.py",
    "scripts/run_cmc_research.py",
    "scripts/smoke_test_rirufold_rcp.py",
    "scripts/smoke_test_rirufold_cmc.py",
    "scripts/smoke_test_server_pipeline.py",
    "scripts/smoke_test_cmc_server_pipeline.py",
    "scripts/smoke_test_cmc_result_analysis.py",
    "scripts/evaluate_rirufold_rcp.py",
    "scripts/diagnose_rirufold_rcp.py",
    "scripts/diagnose_rirufold_cmc.py",
    "scripts/analyze_cmc_server_results.py",
    "scripts/run_server_ablation.sh",
    "scripts/run_cmc_server_ablation.sh",
    "RCP两个研究点_公式代码实验对照.md",
    "创新方法与服务器复现指南.md",
    "CMC创新方法与服务器复现指南.md",
    "题目分析报告.md",
    "术语表格.md",
    "研究记录.md",
)

OPTIONAL_LOCAL_INPUTS = (
    "datasets/NUDT-SIRST/test/text.txt",
    "datasets/NUDT-SIRST/test/images/000001.png",
    "datasets/NUDT-SIRST/test/masks/000001.png",
    "docs/manuscript_zb.pdf",
    "docs/TKDE-2026-08-2894_Proof_hi (1).pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce q1/q2 local gates and mechanism figures")
    parser.add_argument("--skip-pipeline", action="store_true", help="skip temporary checkpoint CLI tests")
    return parser.parse_args()


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, script, *arguments],
        cwd=PROJECT_ROOT,
        env={**os.environ, "MPLCONFIGDIR": str(PROJECT_ROOT / ".mplconfig")},
        check=True,
    )


def file_record(relative_path: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return {
        "path": Path(relative_path).as_posix(),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def normalize_svg_outputs() -> None:
    for path in (PROJECT_ROOT / "figures").glob("*.svg"):
        if "_q1_" not in path.name and "_q2_" not in path.name:
            continue
        svg_text = path.read_text(encoding="utf-8")
        normalized_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
        if normalized_svg != svg_text:
            path.write_text(normalized_svg, encoding="utf-8")


def write_reproduction_manifest() -> None:
    result_paths = sorted(
        path for path in (PROJECT_ROOT / "results").iterdir()
        if path.is_file() and path.name != "复现清单.json"
    )
    figure_paths = sorted(
        path for path in (PROJECT_ROOT / "figures").iterdir()
        if path.is_file() and ("_q1_" in path.name or "_q2_" in path.name)
    )
    output_paths = result_paths + figure_paths
    rcp_summary = json.loads(
        (PROJECT_ROOT / "results" / "run_summary.json").read_text(encoding="utf-8")
    )
    cmc_summary = json.loads(
        (PROJECT_ROOT / "results" / "cmc_run_summary.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "untrained deterministic mechanism validation; not benchmark accuracy",
        "commands": {
            "all_local": ".venv/Scripts/python.exe scripts/reproduce_all.py",
            "all_local_without_cli_pipeline": (
                ".venv/Scripts/python.exe scripts/reproduce_all.py --skip-pipeline"
            ),
            "server_rcp": "bash RiRUFold_ADMM/scripts/run_server_ablation.sh",
            "server_cmc": "bash RiRUFold_ADMM/scripts/run_cmc_server_ablation.sh",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_files": [file_record(path) for path in SOURCE_FILES],
        "optional_local_inputs": [
            file_record(path)
            for path in OPTIONAL_LOCAL_INPUTS
            if (PROJECT_ROOT / path).is_file()
        ],
        "output_files": [
            file_record(path.relative_to(PROJECT_ROOT).as_posix())
            for path in output_paths
        ],
        "rcp_summary": rcp_summary,
        "cmc_summary": cmc_summary,
    }
    destination = PROJECT_ROOT / "results" / "复现清单.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run("scripts/smoke_test_rirufold_rcp.py")
    run("scripts/smoke_test_rirufold_cmc.py")
    if not args.skip_pipeline:
        run("scripts/smoke_test_server_pipeline.py")
        run("scripts/smoke_test_cmc_server_pipeline.py")
        run("scripts/smoke_test_cmc_result_analysis.py")
    run("scripts/run_rcp_research.py", "--seed", "20260910")
    run("scripts/run_cmc_research.py", "--seed", "20260912")
    normalize_svg_outputs()
    write_reproduction_manifest()
    print("Reproduced q1 and q2 local gates, tables, and figures.")


if __name__ == "__main__":
    main()
