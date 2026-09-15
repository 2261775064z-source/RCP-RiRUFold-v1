"""Counterfactual Masked-Completion RiRUFold (CMC-RiRUFold).

CMC changes the background observation model rather than merely changing the
sparse threshold.  A stagewise soft candidate mask excludes likely targets
from direct input fidelity, a row-stochastic annular operator supplies a
target-free side estimate, and two patch-low-rank denoising steps exchange
information with that side estimate before positive-precision consensus.

The implementation follows Section 12 of ``../题目分析报告.md``.  Patch SVT is
used as an explicit scale-conditioned low-rank update; no claim is made that
overlapping patch SVT exactly minimizes the image-domain conditional objective.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rirufold_rcp import (
    BoundedCorrection,
    EPS_SCALE,
    FixedStructureTensor,
    MIN_LOGIT_SCALE,
    MultiScalePatchLowRankEstimator,
    _inverse_softplus,
    _safe_pad,
    _spatial_rms,
    _spatial_standardize,
    positive_soft_threshold,
)


__all__ = [
    "CandidateObservationMask",
    "RowStochasticAnnularPrior",
    "CandidateConditionedWeight",
    "CMCUnfoldStage",
    "CMCRiRUFold",
    "build_cmc_model",
]


EPS_SCORE = 1.0e-6
EPS_SIDE = 1.0e-6
EPS_PRECISION = 1.0e-6
EPS_MASK = 1.0e-6


def _inverse_sigmoid(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    return math.log(probability / (1.0 - probability))


def _precision_raw(value: float) -> float:
    if not 0.25 < value < 4.0:
        raise ValueError("normal precision initialization must be in (0.25, 4)")
    return _inverse_sigmoid((value - 0.25) / 3.75)


class CandidateObservationMask(nn.Module):
    """Unsupervised, stagewise soft mask for unreliable target observations."""

    def __init__(self, stage_index: int, stage_num: int) -> None:
        super().__init__()
        if stage_num < 1 or not 0 <= stage_index < stage_num:
            raise ValueError("invalid stage index or stage count")
        self.stage_index = int(stage_index)
        self.stage_num = int(stage_num)
        mean_kernel = torch.ones(1, 1, 9, 9, dtype=torch.float32) / 81.0
        self.register_buffer("mean_kernel_9", mean_kernel)
        self.score_logits = nn.Parameter(torch.log(torch.tensor([0.60, 0.20, 0.20])))
        self.raw_temperature = nn.Parameter(torch.tensor(0.0))

    def candidate_fraction(self, height: int, width: int) -> float:
        if height * width < 50:
            raise ValueError("CMC requires H*W >= 50 for its candidate-area schedule")
        minimum = max(1.0 / float(height * width), 5.0e-4)
        maximum = 0.02
        progress = self.stage_index / float(max(self.stage_num - 1, 1))
        return maximum - (maximum - minimum) * progress

    def forward(
        self,
        image: torch.Tensor,
        background: torch.Tensor,
        sparse: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        local_mean = F.conv2d(_safe_pad(image, 4), self.mean_kernel_9)
        high_pass = F.relu(image - local_mean)
        decomposition_residual = F.relu(image - background)
        components = torch.stack(
            (
                high_pass / _spatial_rms(high_pass),
                sparse / _spatial_rms(sparse),
                decomposition_residual / _spatial_rms(decomposition_residual),
            ),
            dim=1,
        )
        mixture = torch.softmax(self.score_logits, dim=0).view(1, 3, 1, 1, 1)
        score = (mixture * components).sum(dim=1)

        fraction = self.candidate_fraction(image.shape[-2], image.shape[-1])
        quantile = torch.quantile(
            score.flatten(2), 1.0 - fraction, dim=2, keepdim=True
        ).unsqueeze(-1).detach()
        score_std = score.std(dim=(2, 3), keepdim=True, unbiased=False)
        nonflat_gate = score_std / (score_std + 0.05)
        temperature = (
            0.02 + 0.18 * torch.sigmoid(self.raw_temperature)
        ) * (score_std + EPS_SCORE)
        candidate = nonflat_gate * torch.sigmoid((score - quantile) / temperature)
        candidate = candidate.clamp(0.0, 1.0)
        valid = 1.0 - candidate
        target_fraction = torch.as_tensor(
            fraction, dtype=image.dtype, device=image.device
        ).view(1, 1, 1, 1)
        budget = (
            candidate.mean(dim=(2, 3), keepdim=True) - nonflat_gate * target_fraction
        ).square().mean()
        return {
            "candidate_score": score,
            "candidate": candidate,
            "valid": valid,
            "candidate_quantile": quantile,
            "score_std": score_std,
            "nonflat_gate": nonflat_gate,
            "temperature": temperature,
            "candidate_fraction": target_fraction,
            "budget_loss": budget,
            "score_weights": torch.softmax(self.score_logits, dim=0),
        }


class RowStochasticAnnularPrior(nn.Module):
    """Constant-preserving side prior using fixed 5x5 and 9x9 rings."""

    def __init__(self) -> None:
        super().__init__()
        kernel_5 = torch.ones(5, 5, dtype=torch.float32)
        kernel_5[1:4, 1:4] = 0.0
        kernel_5 /= kernel_5.sum()
        kernel_9 = torch.ones(9, 9, dtype=torch.float32)
        kernel_9[2:7, 2:7] = 0.0
        kernel_9 /= kernel_9.sum()
        self.register_buffer("kernel_5", kernel_5.view(1, 1, 5, 5))
        self.register_buffer("kernel_9", kernel_9.view(1, 1, 9, 9))
        self.branch_bias = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _branch(
        value: torch.Tensor,
        valid: torch.Tensor,
        kernel: torch.Tensor,
        padding: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        support = F.conv2d(_safe_pad(valid, padding), kernel)
        numerator = F.conv2d(_safe_pad(valid * value, padding), kernel) + EPS_SIDE * value
        denominator = support + EPS_SIDE
        return numerator / denominator, denominator

    def forward(
        self, value: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_5, support_5 = self._branch(value, valid, self.kernel_5, 2)
        branch_9, support_9 = self._branch(value, valid, self.kernel_9, 4)
        branches = torch.stack((branch_5, branch_9), dim=1)
        supports = torch.stack((support_5, support_9), dim=1)
        logits = self.branch_bias.view(1, 2, 1, 1, 1) + torch.log(supports)
        weights = torch.softmax(logits, dim=1)
        side = (weights * branches).sum(dim=1)
        return side, branches, weights


class CandidateConditionedWeight(nn.Module):
    """Log-domain structure/sparsity weight conditioned on candidate status."""

    def __init__(self) -> None:
        super().__init__()
        self.structure = FixedStructureTensor()
        self.raw_structure_slope = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.structure_bias = nn.Parameter(torch.tensor(0.0))
        self.raw_sparse_epsilon = nn.Parameter(torch.tensor(_inverse_softplus(0.01)))
        self.raw_candidate_strength = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        background: torch.Tensor,
        sparse: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coherence, energy, _ = self.structure(background)
        slope = F.softplus(self.raw_structure_slope) + EPS_SCORE
        raw_structure = torch.log1p(F.softplus(slope * coherence * energy + self.structure_bias))
        q_background = _spatial_standardize(raw_structure)
        sparse_epsilon = F.softplus(self.raw_sparse_epsilon) + EPS_SCORE
        raw_sparse = -0.5 * torch.log(sparse.square() + sparse_epsilon.square())
        q_sparse = _spatial_standardize(raw_sparse)
        beta = 0.50 * torch.sigmoid(self.raw_candidate_strength)
        centered_candidate = candidate - candidate.mean(dim=(2, 3), keepdim=True)
        log_weight = q_background + q_sparse - beta * centered_candidate
        stable_log_weight = log_weight - log_weight.amax(dim=(2, 3), keepdim=True)
        unnormalized = torch.exp(stable_log_weight)
        weight = unnormalized / (
            unnormalized.mean(dim=(2, 3), keepdim=True) + EPS_SCORE
        )
        return {
            "weight": weight.clamp(0.25, 4.0),
            "q_background": q_background,
            "q_sparse": q_sparse,
            "coherence": coherence,
            "candidate_strength": beta,
        }


class CMCUnfoldStage(nn.Module):
    """One counterfactual masked-completion, ADMM-inspired unfolded stage."""

    def __init__(
        self,
        stage_index: int,
        stage_num: int,
        hidden_channels: int = 24,
        patch_specs: Sequence[Tuple[int, int]] = ((8, 4), (12, 6), (16, 8)),
        inner_steps: int = 2,
        use_side_prior: bool = True,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        if inner_steps not in {1, 2}:
            raise ValueError("CMC preregisters exactly one or two inner steps")
        self.inner_steps = int(inner_steps)
        self.use_side_prior = bool(use_side_prior)
        self.use_mask = bool(use_mask)
        self.masker = CandidateObservationMask(stage_index, stage_num)
        self.side_prior = RowStochasticAnnularPrior()
        self.completion_estimator = MultiScalePatchLowRankEstimator(patch_specs)
        self.raw_precision = nn.Parameter(
            torch.tensor(
                [_precision_raw(2.0), _precision_raw(1.0), _precision_raw(2.0)]
            )
        )
        self.background_correction = BoundedCorrection(4, hidden_channels)
        self.weight_model = CandidateConditionedWeight()
        self.raw_lambda = nn.Parameter(torch.tensor(_inverse_softplus(0.02)))
        self.sparse_correction = BoundedCorrection(4, hidden_channels)
        self.dual_gate = nn.Sequential(
            nn.Conv2d(1, max(4, hidden_channels // 2), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, hidden_channels // 2), 1, 3, padding=1),
        )
        nn.init.zeros_(self.dual_gate[-1].weight)
        nn.init.zeros_(self.dual_gate[-1].bias)
        self.raw_eta = nn.Parameter(torch.tensor(-2.2))

    def _mask_trace(
        self,
        image: torch.Tensor,
        background: torch.Tensor,
        sparse: torch.Tensor,
        mask_override: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not self.use_mask:
            zeros = torch.zeros_like(image)
            ones = torch.ones_like(image)
            scalar_zero = image.new_zeros(())
            return {
                "candidate_score": zeros,
                "candidate": zeros,
                "valid": ones,
                "candidate_quantile": image.new_zeros((image.shape[0], 1, 1, 1)),
                "score_std": image.new_zeros((image.shape[0], 1, 1, 1)),
                "nonflat_gate": image.new_zeros((image.shape[0], 1, 1, 1)),
                "temperature": image.new_zeros((image.shape[0], 1, 1, 1)),
                "candidate_fraction": image.new_zeros((1, 1, 1, 1)),
                "budget_loss": scalar_zero,
                "score_weights": torch.softmax(self.masker.score_logits, dim=0),
            }
        if mask_override is None:
            return self.masker(image, background, sparse)

        trace = {name: value for name, value in mask_override.items()}
        # A fixed-mask ablation freezes the complete stage-0 observation
        # contract. Reusing C^0 while imposing later, smaller area budgets
        # would change two factors at once and unfairly penalize the ablation.
        target_fraction = trace["candidate_fraction"]
        trace["budget_loss"] = (
            trace["candidate"].mean(dim=(2, 3), keepdim=True)
            - trace["nonflat_gate"] * target_fraction
        ).square().mean()
        return trace

    def forward(
        self,
        image: torch.Tensor,
        background: torch.Tensor,
        sparse: torch.Tensor,
        dual: torch.Tensor,
        mu: torch.Tensor,
        mask_override: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        mask_trace = self._mask_trace(image, background, sparse, mask_override)
        candidate = mask_trace["candidate"]
        valid = mask_trace["valid"]
        z_background = image - sparse + dual

        precision = 0.25 + 3.75 * torch.sigmoid(self.raw_precision)
        p_x, p_l = precision[0], precision[1]
        p_a = precision[2] if self.use_side_prior else precision.new_zeros(())

        if self.use_side_prior:
            side, side_branches, side_weights = self.side_prior(background, valid)
            completion = valid * image + candidate * side
        else:
            side = torch.zeros_like(image)
            side_branches = torch.zeros(
                image.shape[0], 2, 1, image.shape[-2], image.shape[-1],
                dtype=image.dtype, device=image.device,
            )
            side_weights = torch.zeros_like(side_branches)
            completion = valid * image + candidate * z_background

        inner_states: List[torch.Tensor] = [completion]
        inner_sides: List[torch.Tensor] = [side]
        scale_backgrounds = completion.unsqueeze(1).expand(-1, 3, -1, -1, -1)
        scale_weights = torch.full_like(scale_backgrounds, 1.0 / 3.0)
        disagreement = torch.zeros_like(image)
        step_size = 1.0 / (p_x + p_l + p_a + EPS_PRECISION)
        completion_mu = torch.ones_like(mu) / step_size
        for _ in range(self.inner_steps):
            gradient = p_x * valid * (completion - image) + p_l * (completion - z_background)
            if self.use_side_prior:
                gradient = gradient + p_a * candidate * (completion - side)
            proposal = completion - step_size * gradient
            completion, scale_backgrounds, scale_weights, disagreement = self.completion_estimator(
                proposal, completion_mu
            )
            if self.use_side_prior:
                side, side_branches, side_weights = self.side_prior(completion, valid)
            inner_states.append(completion)
            inner_sides.append(side)

        denominator = p_x * valid + p_l + p_a * candidate
        numerator = p_x * valid * image + p_l * completion
        if self.use_side_prior:
            numerator = numerator + p_a * candidate * side
        background_bar = numerator / denominator

        background_scale = _spatial_rms(z_background).detach()
        background_delta, gamma_background = self.background_correction(
            background_scale, z_background, background_bar, candidate, side
        )
        background_next = background_bar + background_delta

        z_sparse = image - background_next + dual
        weight_trace = self.weight_model(background_next, sparse, candidate)
        sparse_weight = weight_trace["weight"]
        threshold = (F.softplus(self.raw_lambda) + EPS_SCORE) * sparse_weight / mu.clamp_min(0.05)
        sparse_bar = positive_soft_threshold(z_sparse, threshold)
        sparse_scale = _spatial_rms(z_sparse).detach()
        sparse_delta, gamma_sparse = self.sparse_correction(
            sparse_scale, z_sparse, sparse_bar, sparse_weight, candidate
        )
        sparse_next = F.relu(sparse_bar + sparse_delta)

        primal_residual = image - background_next - sparse_next
        eps_residual = 1.0e-8 * math.sqrt(float(image.shape[-2] * image.shape[-1]))
        primal_norm = torch.linalg.vector_norm(primal_residual, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        dual_norm = mu * torch.linalg.vector_norm(
            sparse_next - sparse, dim=(1, 2, 3)
        ).view(-1, 1, 1, 1)
        alpha = 0.05 + 0.95 * torch.sigmoid(self.dual_gate(primal_residual))
        dual_star = dual + alpha * primal_residual
        eta = 0.25 * torch.sigmoid(self.raw_eta)
        balance = torch.tanh(
            torch.log((primal_norm + eps_residual) / (dual_norm + eps_residual))
        )
        mu_next = (mu * torch.exp(eta * balance)).clamp(0.05, 20.0)
        dual_next = (mu / mu_next) * dual_star

        if self.use_side_prior:
            counterfactual_per_sample = (
                (candidate * (background_next - side).abs()).mean(dim=(1, 2, 3))
                / (candidate.mean(dim=(1, 2, 3)) + EPS_MASK)
            )
            counterfactual_loss = counterfactual_per_sample.mean()
        else:
            counterfactual_loss = image.new_zeros(())

        trace: Dict[str, torch.Tensor] = {
            **mask_trace,
            "background": background_next,
            "sparse_intensity": sparse_next,
            "z_background": z_background,
            "side_prior": side,
            "side_branches": side_branches,
            "side_weights": side_weights,
            "side_enabled": image.new_tensor(float(self.use_side_prior)),
            "completion_inner": torch.stack(inner_states, dim=1),
            "completion_side_inner": torch.stack(inner_sides, dim=1),
            "background_bar": background_bar,
            "precision": torch.stack((p_x, p_l, p_a)),
            "precision_denominator": denominator,
            "completion_step": step_size,
            "scale_backgrounds": scale_backgrounds,
            "scale_weights": scale_weights,
            "scale_disagreement": disagreement,
            "w_cmc": sparse_weight,
            "q_background": weight_trace["q_background"],
            "q_sparse": weight_trace["q_sparse"],
            "candidate_strength": weight_trace["candidate_strength"],
            "sparse_threshold": threshold,
            "sparse_bar": sparse_bar,
            "primal_residual": primal_residual,
            "primal_norm": primal_norm,
            "dual_norm": dual_norm,
            "mu": mu,
            "mu_next": mu_next,
            "alpha": alpha,
            "background_delta": background_delta,
            "sparse_delta": sparse_next - sparse_bar,
            "background_correction_bound": gamma_background * background_scale,
            "sparse_correction_bound": gamma_sparse * sparse_scale,
            "background_scale": background_scale,
            "sparse_scale": sparse_scale,
            "counterfactual_loss": counterfactual_loss,
        }
        return background_next, sparse_next, dual_next, mu_next, trace


class CMCRiRUFold(nn.Module):
    """Multi-stage CMC network with separate physical sparse and logit outputs."""

    def __init__(
        self,
        stage_num: int = 5,
        hidden_channels: int = 24,
        patch_specs: Sequence[Tuple[int, int]] = ((8, 4), (12, 6), (16, 8)),
        inner_steps: int = 2,
        use_side_prior: bool = True,
        use_mask: bool = True,
        fixed_mask: bool = False,
    ) -> None:
        super().__init__()
        if stage_num < 1:
            raise ValueError("stage_num must be positive")
        self.stage_num = int(stage_num)
        self.fixed_mask = bool(fixed_mask)
        self.stages = nn.ModuleList(
            [
                CMCUnfoldStage(
                    stage_index=index,
                    stage_num=stage_num,
                    hidden_channels=hidden_channels,
                    patch_specs=patch_specs,
                    inner_steps=inner_steps,
                    use_side_prior=use_side_prior,
                    use_mask=use_mask,
                )
                for index in range(stage_num)
            ]
        )
        self.raw_logit_gain = nn.Parameter(torch.tensor(_inverse_softplus(4.0)))
        self.raw_logit_bias = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))

    def forward(
        self,
        image: torch.Tensor,
        return_aux: bool = False,
        return_trace: bool = False,
    ):
        if return_aux and return_trace:
            raise ValueError("return_aux and return_trace are mutually exclusive")
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("CMCRiRUFold expects normalized Bx1xHxW input")
        if image.shape[-2] * image.shape[-1] < 50:
            raise ValueError("CMCRiRUFold requires H*W >= 50")

        background = image.clone()
        sparse = torch.zeros_like(image)
        dual = torch.zeros_like(image)
        centered = image - image.mean(dim=(2, 3), keepdim=True)
        mu = (5.0 * torch.sqrt(
            centered.square().mean(dim=(2, 3), keepdim=True) + EPS_SCALE**2
        )).clamp(0.05, 20.0)
        traces: List[Dict[str, torch.Tensor]] = []
        prox_terms: List[torch.Tensor] = []
        budget_terms: List[torch.Tensor] = []
        counterfactual_terms: List[torch.Tensor] = []
        cached_mask: Optional[Dict[str, torch.Tensor]] = None

        for index, stage in enumerate(self.stages):
            override = cached_mask if self.fixed_mask and index > 0 else None
            background, sparse, dual, mu, trace = stage(
                image, background, sparse, dual, mu, mask_override=override
            )
            if self.fixed_mask and index == 0:
                cached_mask = {
                    name: trace[name]
                    for name in (
                        "candidate_score", "candidate", "valid", "candidate_quantile",
                        "score_std", "nonflat_gate", "temperature", "candidate_fraction",
                        "score_weights",
                    )
                }
            background_ratio = trace["background_delta"].abs().mean(dim=(1, 2, 3)) / (
                trace["background_scale"].reshape(-1) + EPS_SCALE
            )
            sparse_ratio = trace["sparse_delta"].abs().mean(dim=(1, 2, 3)) / (
                trace["sparse_scale"].reshape(-1) + EPS_SCALE
            )
            prox_terms.append((background_ratio + sparse_ratio).mean())
            budget_terms.append(trace["budget_loss"])
            counterfactual_terms.append(trace["counterfactual_loss"])
            traces.append(trace)

        logit_scale = torch.sqrt(
            centered.square().mean(dim=(2, 3), keepdim=True) + EPS_SCALE**2
        ).clamp_min(MIN_LOGIT_SCALE).detach()
        gain = F.softplus(self.raw_logit_gain) + EPS_SCALE
        bias = F.softplus(self.raw_logit_bias) + EPS_SCALE
        logits = gain * (sparse / logit_scale - bias)

        if return_trace:
            detached_trace = [
                {name: value.detach() for name, value in stage.items()} for stage in traces
            ]
            return background, logits, detached_trace
        if return_aux:
            aux = {
                "sparse_intensity": sparse,
                "prox_loss": torch.stack(prox_terms).mean(),
                "mask_budget_loss": torch.stack(budget_terms).mean(),
                "counterfactual_loss": torch.stack(counterfactual_terms).mean(),
                "final_mu": mu,
                "final_dual": dual,
            }
            return background, logits, aux
        return background, logits


def build_cmc_model(name: str = "rirufold_cmc", **kwargs) -> CMCRiRUFold:
    """Build the main CMC model or one preregistered mechanism ablation."""
    configurations = {
        "rirufold_cmc": {},
        "rirufold_cmc_nomask": {"use_mask": False},
        "rirufold_cmc_noside": {"use_side_prior": False},
        "rirufold_cmc_onepass": {"inner_steps": 1},
        "rirufold_cmc_fixedmask": {"fixed_mask": True},
    }
    if name not in configurations:
        raise ValueError("Unknown CMC-RiRUFold variant: {}".format(name))
    selected = dict(configurations[name])
    selected.update(kwargs)
    return CMCRiRUFold(**selected)
