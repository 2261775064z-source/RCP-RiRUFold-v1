"""P1 numerical gate for CMC-RiRUFold.

This validates the second scheme's mechanism and numerical contracts on CPU.
It is not a trained detection benchmark or an accuracy claim.
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

from RiRUFold_ADMM.models.rirufold_cmc import (  # noqa: E402
    RowStochasticAnnularPrior,
    build_cmc_model,
)


PATCH_SPECS = ((4, 2), (6, 3), (8, 4))


def _structured_batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(29)
    image = 0.10 + 0.025 * torch.rand((2, 1, 28, 26), generator=generator)
    image[:, :, :, 13:] += 0.13
    image[0, 0, 8:11, 6:9] += 0.63
    image[1, 0, 17:20, 18:21] += 0.54
    return image.clamp(0.0, 1.0)


def _build(name: str, stage_num: int = 1, hidden_channels: int = 6):
    return build_cmc_model(
        name,
        stage_num=stage_num,
        hidden_channels=hidden_channels,
        patch_specs=PATCH_SPECS,
    )


def _assert_finite_gradients(model: torch.nn.Module) -> int:
    count = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), "non-finite parameter gradient"
            count += 1
    assert count > 0, "no parameter received a gradient"
    return count


def main() -> None:
    torch.manual_seed(20260912)
    torch.set_num_threads(1)
    results = {}

    # The annular prior must be an actual row-stochastic contraction once V is fixed.
    prior = RowStochasticAnnularPrior().eval()
    valid = torch.rand(2, 1, 21, 23)
    constant = torch.full_like(valid, 0.37)
    with torch.no_grad():
        restored, _, weights = prior(constant, valid)
        signed = 2.0 * torch.rand_like(valid) - 1.0
        contracted, _, _ = prior(signed, valid)
    constant_error = float((restored - constant).abs().max())
    assert constant_error < 2.0e-6, constant_error
    assert float(contracted.abs().max()) <= float(signed.abs().max()) + 2.0e-6
    assert float((weights.sum(dim=1) - 1.0).abs().max()) < 1.0e-6
    results["side_constant_max_abs_error"] = constant_error
    results["side_linf_ratio"] = float(contracted.abs().max() / signed.abs().max())

    # The preregistered area schedule has fixed endpoints and rejects tiny fields.
    schedule_model = _build("rirufold_cmc", stage_num=5)
    first_fraction = schedule_model.stages[0].masker.candidate_fraction(40, 40)
    last_fraction = schedule_model.stages[-1].masker.candidate_fraction(40, 40)
    assert abs(first_fraction - 0.02) < 1.0e-12
    assert abs(last_fraction - max(1.0 / 1600.0, 5.0e-4)) < 1.0e-12
    try:
        schedule_model.stages[0].masker.candidate_fraction(7, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("H*W < 50 was not rejected")
    results["candidate_fraction_first"] = first_fraction
    results["candidate_fraction_last"] = last_fraction

    # Flat input must not create the otherwise-trivial 0.5 candidate mask.
    flat_model = _build("rirufold_cmc")
    flat = torch.full((1, 1, 24, 24), 0.2)
    with torch.no_grad():
        _, _, flat_trace = flat_model(flat, return_trace=True)
    flat_candidate_max = float(flat_trace[0]["candidate"].max())
    assert flat_candidate_max == 0.0, flat_candidate_max
    results["flat_candidate_max"] = flat_candidate_max

    # Main forward/backward contract, including both completion exchanges.
    image = _structured_batch().requires_grad_(True)
    model = _build("rirufold_cmc")
    background, logits, aux = model(image, return_aux=True)
    sparse = aux["sparse_intensity"]
    decomposition = (image - background - sparse).abs().mean()
    loss = (
        logits.square().mean()
        + 0.05 * decomposition
        + 0.005 * aux["prox_loss"]
        + 0.1 * aux["mask_budget_loss"]
        + 0.01 * aux["counterfactual_loss"]
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(image.grad).all()
    gradient_count = _assert_finite_gradients(model)
    assert sparse.data_ptr() != logits.data_ptr()
    results["backward_finite_parameter_count"] = gradient_count
    results["input_gradient_max_abs"] = float(image.grad.abs().max())
    results["decomposition_loss"] = float(decomposition.detach())

    model.eval()
    with torch.no_grad():
        _, _, trace = model(image.detach(), return_trace=True)
    state = trace[0]
    assert state["completion_inner"].shape[1] == 3
    first_change = float(
        (state["completion_inner"][:, 1] - state["completion_inner"][:, 0]).abs().max()
    )
    second_change = float(
        (state["completion_inner"][:, 2] - state["completion_inner"][:, 1]).abs().max()
    )
    assert first_change > 1.0e-7 and second_change > 1.0e-7
    assert ((state["candidate"] >= 0.0) & (state["candidate"] <= 1.0)).all()
    assert torch.allclose(state["candidate"] + state["valid"], torch.ones_like(state["valid"]))
    assert ((state["precision"] > 0.25) & (state["precision"] < 4.0)).all()
    assert (state["precision_denominator"] > 0.0).all()
    assert ((state["w_cmc"] >= 0.25) & (state["w_cmc"] <= 4.0)).all()
    assert (state["background_delta"].abs() <= state["background_correction_bound"] + 1.0e-6).all()
    assert (state["sparse_delta"].abs() <= state["sparse_correction_bound"] + 1.0e-6).all()
    results["first_inner_max_change"] = first_change
    results["second_inner_max_change"] = second_change
    results["minimum_precision_denominator"] = float(state["precision_denominator"].min())

    # The synthetic bright target should be trusted less as a direct background observation.
    candidate = state["candidate"][0, 0]
    target_candidate = float(candidate[8:11, 6:9].mean())
    background_candidate = float(candidate[:5, :5].mean())
    assert target_candidate > background_candidate, (target_candidate, background_candidate)
    results["target_candidate_mean"] = target_candidate
    results["background_candidate_mean"] = background_candidate

    # Full J=2 performs initial side construction plus one refresh per update.
    counted = _build("rirufold_cmc").eval()
    original_side_forward = counted.stages[0].side_prior.forward
    side_call_count = [0]

    def counted_side_call(*args, **kwargs):
        side_call_count[0] += 1
        return original_side_forward(*args, **kwargs)

    counted.stages[0].side_prior.forward = counted_side_call
    with torch.no_grad():
        counted(image.detach()[0:1])
    assert side_call_count[0] == 3, side_call_count[0]
    results["full_stage_side_call_count"] = side_call_count[0]

    # Each ablation has a falsifiable trace contract.
    nomask = _build("rirufold_cmc_nomask").eval()
    with torch.no_grad():
        _, _, nomask_trace = nomask(image.detach()[0:1], return_trace=True)
    assert float(nomask_trace[0]["candidate"].max()) == 0.0
    assert float(nomask_trace[0]["budget_loss"]) == 0.0
    assert float(nomask_trace[0]["counterfactual_loss"]) == 0.0

    noside = _build("rirufold_cmc_noside").eval()

    def forbidden_side_call(*_args, **_kwargs):
        raise AssertionError("noside called the side prior")

    noside.stages[0].side_prior.forward = forbidden_side_call
    with torch.no_grad():
        _, _, noside_trace = noside(image.detach()[0:1], return_trace=True)
    assert float(noside_trace[0]["side_enabled"]) == 0.0
    assert float(noside_trace[0]["precision"][2]) == 0.0
    assert float(noside_trace[0]["side_prior"].abs().max()) == 0.0
    assert float(noside_trace[0]["counterfactual_loss"]) == 0.0

    onepass = _build("rirufold_cmc_onepass").eval()
    with torch.no_grad():
        _, _, onepass_trace = onepass(image.detach()[0:1], return_trace=True)
    assert onepass_trace[0]["completion_inner"].shape[1] == 2

    fixed = _build("rirufold_cmc_fixedmask", stage_num=2).eval()
    with torch.no_grad():
        _, _, fixed_trace = fixed(image.detach()[0:1], return_trace=True)
    fixed_error = float((fixed_trace[0]["candidate"] - fixed_trace[1]["candidate"]).abs().max())
    assert fixed_error == 0.0
    assert torch.equal(
        fixed_trace[0]["candidate_fraction"], fixed_trace[1]["candidate_fraction"]
    ), "fixedmask must freeze the stage-0 budget together with C^0/V^0"
    results["fixedmask_stage_max_abs_error"] = fixed_error
    results["noside_side_enabled"] = float(noside_trace[0]["side_enabled"])

    # Per-sample reductions and quantiles may not leak companion-image statistics.
    sample = image.detach()[0:1]
    companion = torch.flip(image.detach()[1:2], dims=(-1,)) * 0.31
    with torch.no_grad():
        single_background, single_logits = model(sample)
        pair_background, pair_logits = model(torch.cat((sample, companion), dim=0))
    batch_background_error = float((single_background - pair_background[0:1]).abs().max())
    batch_logit_error = float((single_logits - pair_logits[0:1]).abs().max())
    assert batch_background_error < 2.0e-5, batch_background_error
    assert batch_logit_error < 2.0e-5, batch_logit_error
    results["batch_invariance_background_max_abs_error"] = batch_background_error
    results["batch_invariance_logits_max_abs_error"] = batch_logit_error

    # Flat repeated-singular-value backward remains finite through both inner steps.
    flat_grad = flat.clone().requires_grad_(True)
    flat_train_model = _build("rirufold_cmc", hidden_channels=4)
    flat_background, flat_logits, flat_aux = flat_train_model(flat_grad, return_aux=True)
    flat_loss = (
        flat_background.mean() + flat_logits.square().mean()
        + flat_aux["prox_loss"] + flat_aux["mask_budget_loss"]
        + flat_aux["counterfactual_loss"]
    )
    flat_loss.backward()
    assert torch.isfinite(flat_grad.grad).all()
    _assert_finite_gradients(flat_train_model)
    results["flat_input_gradient_max_abs"] = float(flat_grad.grad.abs().max())

    # Decode one real local NUDT-SIRST image and export a complete trace.
    nudt_images = sorted(
        (PACKAGE_ROOT / "datasets" / "NUDT-SIRST" / "test" / "images").glob("*.png")
    )
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
    print("CMC-RiRUFold P1 smoke test passed.")


if __name__ == "__main__":
    main()
