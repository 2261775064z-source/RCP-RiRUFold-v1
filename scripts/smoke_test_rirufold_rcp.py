"""P1 numerical gate for RCP-RiRUFold.

This is a mechanism and implementation test, not an accuracy claim.  It uses
small CPU tensors plus one resized local NUDT-SIRST image so it can run before
server training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_RESAMPLING = getattr(Image, "Resampling", Image)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RiRUFold_ADMM.models.rirufold_rcp import (  # noqa: E402
    RCPRiRUFold,
    build_rcp_model,
    median_deviation_fusion,
    overlapping_patch_svt,
)


def _structured_batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    image = 0.12 + 0.02 * torch.rand((2, 1, 24, 22), generator=generator)
    image[:, :, :, 11:] += 0.18
    image[0, 0, 7:9, 5:7] += 0.55
    image[1, 0, 15:17, 16:18] += 0.48
    return image.clamp(0.0, 1.0)


def _assert_finite_gradients(model: torch.nn.Module) -> int:
    count = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), "non-finite parameter gradient"
            count += 1
    assert count > 0, "no parameter received a gradient"
    return count


def main() -> None:
    torch.manual_seed(13)
    torch.set_num_threads(1)
    results = {}

    # Exact fold/unfold reconstruction at zero threshold on a non-divisible size.
    constant = torch.full((1, 1, 23, 29), 0.37)
    reconstructed = overlapping_patch_svt(
        constant, torch.zeros((1, 1, 1, 1)), patch_size=8, stride=4
    )
    fold_error = float((reconstructed - constant).abs().max())
    assert fold_error < 1.0e-5, fold_error
    results["zero_threshold_fold_max_abs_error"] = fold_error

    # Research point 1 contract: increasing one branch's median-referenced
    # deviation must strictly reduce that branch's bias-free fusion weight.
    scale = torch.ones((1, 1, 1, 1))
    kappa = torch.tensor(1.0)
    mild = torch.tensor([0.0, 0.0, 0.5]).view(1, 3, 1, 1, 1)
    severe = torch.tensor([0.0, 0.0, 1.5]).view(1, 3, 1, 1, 1)
    _, mild_weights, _ = median_deviation_fusion(mild, scale, kappa)
    _, severe_weights, _ = median_deviation_fusion(severe, scale, kappa)
    assert severe_weights[0, 2, 0, 0, 0] < mild_weights[0, 2, 0, 0, 0]
    results["outlier_weight_mild"] = float(mild_weights[0, 2, 0, 0, 0])
    results["outlier_weight_severe"] = float(severe_weights[0, 2, 0, 0, 0])

    image = _structured_batch().requires_grad_(True)
    model = build_rcp_model("rirufold_rcp", stage_num=1, hidden_channels=8)
    background, logits, aux = model(image, return_aux=True)
    sparse = aux["sparse_intensity"]
    decomposition_loss = (image - background - sparse).abs().mean()
    loss = logits.square().mean() + 0.05 * decomposition_loss + 0.005 * aux["prox_loss"]
    loss.backward()
    assert torch.isfinite(background).all() and torch.isfinite(logits).all()
    assert torch.isfinite(sparse).all() and torch.isfinite(loss)
    gradient_count = _assert_finite_gradients(model)
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert sparse.data_ptr() != logits.data_ptr(), "physical sparse state and logits are aliased"
    assert aux["final_mu"].shape == (2, 1, 1, 1)
    results["backward_finite_parameter_count"] = gradient_count
    results["input_gradient_max_abs"] = float(image.grad.abs().max())
    results["decomposition_loss"] = float(decomposition_loss.detach())

    model.eval()
    with torch.no_grad():
        _, _, trace = model(image.detach(), return_trace=True)
    assert len(trace) == 1 and trace[0]["scale_backgrounds"].shape[1] == 3
    state = trace[0]
    assert not hasattr(model.stages[0].background_estimator, "branch_bias")
    assert torch.allclose(state["scale_weights"].sum(dim=1), torch.ones_like(state["background"]))
    assert ((state["w_rcp"] >= 0.25) & (state["w_rcp"] <= 4.0)).all()
    assert ((state["alpha"] > 0.05) & (state["alpha"] < 1.0)).all()
    assert ((state["mu_next"] >= 0.05) & (state["mu_next"] <= 20.0)).all()
    assert (state["background_delta"].abs() <= state["background_correction_bound"] + 1.0e-6).all()
    assert (state["sparse_delta"].abs() <= state["sparse_correction_bound"] + 1.0e-6).all()

    # Batch-composition invariance: the same image must not inherit a companion's scale.
    sample = image.detach()[0:1]
    companion = torch.flip(image.detach()[1:2], dims=(-1,)) * 0.35
    with torch.no_grad():
        single_background, single_logits = model(sample)
        pair_background, pair_logits = model(torch.cat([sample, companion], dim=0))
    batch_background_error = float((single_background - pair_background[0:1]).abs().max())
    batch_logit_error = float((single_logits - pair_logits[0:1]).abs().max())
    assert batch_background_error < 2.0e-5, batch_background_error
    assert batch_logit_error < 2.0e-5, batch_logit_error
    results["batch_invariance_background_max_abs_error"] = batch_background_error
    results["batch_invariance_logits_max_abs_error"] = batch_logit_error

    # A positive logit derivative verifies the stopped-scale monotone readout contract.
    sparse_probe = sparse.detach().clone().requires_grad_(True)
    input_centered = image.detach() - image.detach().mean((2, 3), keepdim=True)
    scale_probe = torch.sqrt(input_centered.square().mean((2, 3), keepdim=True) + 1.0e-12).clamp_min(0.01).detach()
    gain = torch.nn.functional.softplus(model.raw_logit_gain) + 1.0e-6
    bias = torch.nn.functional.softplus(model.raw_logit_bias) + 1.0e-6
    probe_logits = gain * (sparse_probe / scale_probe - bias)
    probe_gradient = torch.autograd.grad(probe_logits.sum(), sparse_probe)[0]
    assert (probe_gradient > 0).all()
    slope_upper_bound = float(gain.detach() / 0.01)
    assert float(probe_gradient.max()) <= slope_upper_bound + 1.0e-4
    results["minimum_logit_derivative"] = float(probe_gradient.min())
    results["maximum_logit_derivative"] = float(probe_gradient.max())
    results["logit_derivative_upper_bound"] = slope_upper_bound

    # Repeated singular values are common in flat backgrounds.  The exact
    # forward / stabilized-backward SVT path must keep this case finite.
    flat = torch.full((1, 1, 18, 18), 0.2, requires_grad=True)
    flat_model = build_rcp_model("rirufold_rcp", stage_num=1, hidden_channels=4)
    flat_background, flat_logits, flat_aux = flat_model(flat, return_aux=True)
    flat_loss = flat_background.mean() + flat_logits.square().mean() + flat_aux["prox_loss"]
    flat_loss.backward()
    assert flat.grad is not None and torch.isfinite(flat.grad).all()
    _assert_finite_gradients(flat_model)
    results["flat_input_gradient_max_abs"] = float(flat.grad.abs().max())

    # The proximal deviation reduction must preserve per-sample semantics.
    regression_model = build_rcp_model("rirufold_rcp", stage_num=1, hidden_channels=4).eval()
    with torch.no_grad():
        for stage in regression_model.stages:
            stage.background_correction.body[-1].bias.fill_(0.15)
            stage.sparse_correction.body[-1].bias.fill_(0.15)
        pair_aux = regression_model(image.detach(), return_aux=True)[2]
        first_aux = regression_model(image.detach()[0:1], return_aux=True)[2]
        second_aux = regression_model(image.detach()[1:2], return_aux=True)[2]
    expected_pair_prox = 0.5 * (first_aux["prox_loss"] + second_aux["prox_loss"])
    prox_batch_error = float((pair_aux["prox_loss"] - expected_pair_prox).abs())
    assert prox_batch_error < 1.0e-6, prox_batch_error
    results["prox_loss_batch_reduction_error"] = prox_batch_error

    # Registered ablations must all build and preserve the two-output contract.
    for variant in (
        "rirufold_rcp_product",
        "rirufold_rcp_fixedgate",
        "rirufold_rcp_mean",
        "rirufold_rcp_nofeedback",
        "rirufold_rcp_blob",
        "rirufold_rcp_global",
    ):
        candidate = build_rcp_model(variant, stage_num=1, hidden_channels=4)
        with torch.no_grad():
            candidate_background, candidate_logits = candidate(sample[:, :, :18, :18])
        assert candidate_background.shape == candidate_logits.shape == (1, 1, 18, 18)

    with torch.no_grad():
        mean_trace = build_rcp_model(
            "rirufold_rcp_mean", stage_num=1, hidden_channels=4
        )(sample[:, :, :18, :18], return_trace=True)[2][0]
        fixed_trace = build_rcp_model(
            "rirufold_rcp_fixedgate", stage_num=1, hidden_channels=4
        )(sample[:, :, :18, :18], return_trace=True)[2][0]
    assert torch.allclose(
        mean_trace["scale_weights"],
        torch.full_like(mean_trace["scale_weights"], 1.0 / 3.0),
    )
    assert torch.allclose(
        fixed_trace["structure_gate"], torch.full_like(fixed_trace["structure_gate"], 0.5)
    )

    # One real local image exercises decoding and the complete trace path.  Resize
    # is intentional here: this gate validates the chain, not dataset accuracy.
    nudt_images = sorted((PACKAGE_ROOT / "datasets" / "NUDT-SIRST" / "test" / "images").glob("*.png"))
    assert nudt_images, "No local NUDT-SIRST test image found"
    pil_image = Image.open(nudt_images[0]).convert("L").resize((40, 40), _RESAMPLING.BILINEAR)
    real_tensor = torch.from_numpy(np.asarray(pil_image, dtype=np.float32) / 255.0)[None, None]
    with torch.no_grad():
        real_background, real_logits, real_trace = model(real_tensor, return_trace=True)
    assert real_background.shape == real_logits.shape == real_tensor.shape
    assert torch.isfinite(real_trace[0]["primal_residual"]).all()
    results["nudt_trace_image"] = str(nudt_images[0].relative_to(PACKAGE_ROOT))
    results["nudt_trace_resize"] = [40, 40]

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("RCP-RiRUFold P1 smoke test passed.")


if __name__ == "__main__":
    main()
