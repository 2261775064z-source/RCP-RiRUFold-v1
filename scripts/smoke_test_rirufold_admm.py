"""CPU-only shape and state-consistency test for RiRFoldADMM.

This is not an accuracy test and does not need a dataset or trained weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.rirufold_admm import RiRFoldADMM


def main():
    torch.manual_seed(7)
    image = torch.rand(1, 1, 32, 32)
    model = RiRFoldADMM(stage_num=2, hidden_channels=8, use_svt=True)
    background, sparse, trace = model(image, return_trace=True)

    assert background.shape == image.shape
    assert sparse.shape == image.shape
    assert len(trace) == 2
    for index, state in enumerate(trace, start=1):
        assert state["w_ls"].shape == image.shape
        assert state["w_se"].shape == image.shape
        assert state["w_lse"].shape == image.shape
        assert torch.isfinite(state["primal_residual"]).all()
        assert torch.isfinite(state["mu"])
        assert (state["alpha"] > 0).all() and (state["alpha"] <= 1).all()
        print(
            f"Stage {index}: primal={state['primal_norm'].item():.6f}, "
            f"dual={state['dual_norm'].item():.6f}, mu={state['mu'].item():.6f}, "
            f"alpha_mean={state['alpha'].mean().item():.6f}"
        )
    print("RiRFoldADMM CPU smoke test passed.")


if __name__ == "__main__":
    main()
