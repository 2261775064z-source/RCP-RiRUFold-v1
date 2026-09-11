"""Reliability-Calibrated Patch-Consensus RiRUFold (RCP-RiRUFold).

The implementation follows the contract in ``../题目分析报告.md``:

* three overlapping patch-SVT denoising anchors (or a global-SVT ablation),
* reliability-calibrated log-domain sparse weights,
* hard-bounded learned corrections around explicit anchors,
* per-sample residual-balanced penalties with scaled-dual rescaling, and
* a strict separation between physical sparse intensity and segmentation logits.

Patch-SVT is deliberately described as a denoising anchor.  Overlapping patch
lifting is not orthogonal, so this module does not claim that fold(SVT(unfold))
is the exact image-domain proximal map of a patch nuclear-norm objective.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "RCPRiRUFold",
    "RCPUnfoldStage",
    "FixedStructureTensor",
    "overlapping_patch_svt",
    "build_rcp_model",
]


EPS_SCALE = 1.0e-6
EPS_COVERAGE = 1.0e-6
EPS_STANDARDIZE = 1.0e-3
EPS_TENSOR = 1.0e-8
MIN_LOGIT_SCALE = 1.0e-2


def _inverse_softplus(value: float) -> float:
    """Return x such that softplus(x) is approximately ``value``."""
    return math.log(math.expm1(value))


def _safe_pad(value: torch.Tensor, amount: int) -> torch.Tensor:
    """Reflect-pad when legal and use deterministic replicate padding otherwise."""
    if amount == 0:
        return value
    mode = "reflect" if value.shape[-2] > amount and value.shape[-1] > amount else "replicate"
    return F.pad(value, (amount, amount, amount, amount), mode=mode)


def _spatial_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.square().mean(dim=(2, 3), keepdim=True) + EPS_SCALE**2)


def _spatial_standardize(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=(2, 3), keepdim=True)
    scale = torch.sqrt(centered.square().mean(dim=(2, 3), keepdim=True) + EPS_STANDARDIZE**2)
    return centered / scale


def positive_soft_threshold(value: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Exact nonnegative weighted-l1 proximal anchor for bright targets."""
    return F.relu(value - threshold)


def signed_soft_threshold(value: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Signed ablation for datasets containing both bright and dark targets."""
    return torch.sign(value) * F.relu(torch.abs(value) - threshold)


def _svt_matrix(matrix: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """SVT with an exact forward value and a stabilized training derivative.

    PyTorch's SVD backward is undefined at repeated singular values, which are
    common for flat infrared patches.  During training, the returned forward
    value is still exact SVT(matrix); only its backward surrogate uses a small,
    deterministic, scale-aware diagonal perturbation to split the spectrum.
    """

    def reconstruct(source: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        u, singular, vh = torch.linalg.svd(source, full_matrices=False)
        shrunk = F.relu(singular - level)
        return (u * shrunk.unsqueeze(-2)) @ vh

    if not torch.is_grad_enabled() or not (matrix.requires_grad or threshold.requires_grad):
        return reconstruct(matrix, threshold)

    rows, columns = matrix.shape[-2:]
    rank = min(rows, columns)
    diagonal = torch.zeros((rows, columns), dtype=matrix.dtype, device=matrix.device)
    indices = torch.arange(rank, device=matrix.device)
    diagonal[indices, indices] = torch.linspace(
        1.0, 2.0, rank, dtype=matrix.dtype, device=matrix.device
    )
    matrix_scale = torch.sqrt(matrix.detach().square().mean(dim=(-2, -1), keepdim=True)).clamp_min(0.01)
    stabilized = matrix + 1.0e-3 * matrix_scale * diagonal.view(1, rows, columns)
    surrogate = reconstruct(stabilized, threshold)
    with torch.no_grad():
        exact = reconstruct(matrix.detach(), threshold.detach())
    return surrogate + (exact - surrogate).detach()


def overlapping_patch_svt(
    value: torch.Tensor,
    threshold: torch.Tensor,
    patch_size: int,
    stride: int,
) -> torch.Tensor:
    """Apply batched SVT to one overlapping patch matrix per image.

    ``value`` is Bx1xHxW.  The lifted matrix has shape Bx(p^2)xL and the
    singular-value threshold has shape Bx1, preventing cross-batch broadcast.
    Fold coverage is explicitly normalized before the original field is cropped.
    """
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("overlapping_patch_svt expects Bx1xHxW input")
    if patch_size < 2 or stride < 1 or stride > patch_size // 2:
        raise ValueError("Require patch_size >= 2 and 1 <= stride <= patch_size // 2")

    batch, _, height, width = value.shape
    pad = patch_size // 2
    padded = _safe_pad(value, pad)
    lifted = F.unfold(padded, kernel_size=patch_size, stride=stride, padding=0)
    per_sample_threshold = threshold.reshape(batch, 1)
    restored = _svt_matrix(lifted, per_sample_threshold)

    output_size = (padded.shape[-2], padded.shape[-1])
    folded = F.fold(restored, output_size=output_size, kernel_size=patch_size, stride=stride)
    coverage_patches = torch.ones_like(lifted)
    coverage = F.fold(
        coverage_patches,
        output_size=output_size,
        kernel_size=patch_size,
        stride=stride,
    )
    folded = folded / coverage.clamp_min(EPS_COVERAGE)
    return folded[:, :, pad : pad + height, pad : pad + width]


def global_svt(value: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Full-image matrix SVT used only by the registered global ablation."""
    matrix = value[:, 0]
    restored = _svt_matrix(matrix, threshold.reshape(value.shape[0], 1))
    return restored.unsqueeze(1)


class FixedStructureTensor(nn.Module):
    """Deterministic Sobel/binomial structure-tensor contract."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        binomial = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
        smooth = torch.outer(binomial, binomial).view(1, 1, 5, 5) / 256.0
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2).contiguous())
        self.register_buffer("binomial_5", smooth)
        self.register_buffer("laplacian_4", laplacian)

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded = _safe_pad(value, 1)
        grad_x = F.conv2d(padded, self.sobel_x)
        grad_y = F.conv2d(padded, self.sobel_y)

        jxx = F.conv2d(_safe_pad(grad_x.square(), 2), self.binomial_5)
        jxy = F.conv2d(_safe_pad(grad_x * grad_y, 2), self.binomial_5)
        jyy = F.conv2d(_safe_pad(grad_y.square(), 2), self.binomial_5)
        trace = jxx + jyy
        discriminant = (jxx - jyy).square() + 4.0 * jxy.square()
        eigen_gap = (torch.sqrt(discriminant + EPS_TENSOR**2) - EPS_TENSOR).clamp_min(0.0)
        coherence = (eigen_gap / (trace + EPS_TENSOR)).clamp(0.0, 1.0)
        energy = trace / (trace.mean(dim=(2, 3), keepdim=True) + EPS_TENSOR)
        return coherence, energy, trace

    def positive_blob_response(self, value: torch.Tensor) -> torch.Tensor:
        laplacian = F.conv2d(_safe_pad(value, 1), self.laplacian_4)
        coherence, _, _ = self(value)
        return F.relu(-laplacian) * (1.0 - coherence)


class BoundedCorrection(nn.Module):
    """Learned residual with an explicit per-pixel hard magnitude bound."""

    def __init__(self, in_channels: int, hidden_channels: int, gamma_max: float = 0.20) -> None:
        super().__init__()
        self.gamma_max = float(gamma_max)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.raw_gamma = nn.Parameter(torch.tensor(-2.0))

    def forward(self, local_scale: torch.Tensor, *inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma_max * torch.sigmoid(self.raw_gamma)
        delta = gamma * local_scale * torch.tanh(self.body(torch.cat(inputs, dim=1)))
        return delta, gamma


class PatchConsensusAnchor(nn.Module):
    """Three-scale robust fusion of overlapping patch-SVT denoising anchors."""

    def __init__(self, patch_specs: Sequence[Tuple[int, int]]) -> None:
        super().__init__()
        if len(patch_specs) != 3:
            raise ValueError("The RCP contract requires exactly three patch scales")
        self.patch_specs = tuple((int(p), int(s)) for p, s in patch_specs)
        self.raw_tau = nn.Parameter(
            torch.tensor([_inverse_softplus(0.025), _inverse_softplus(0.035), _inverse_softplus(0.045)])
        )
        self.branch_bias = nn.Parameter(torch.zeros(3))
        self.raw_kappa = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))

    def forward(
        self, value: torch.Tensor, mu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = F.softplus(self.raw_tau) + EPS_SCALE
        anchors: List[torch.Tensor] = []
        for index, (patch_size, stride) in enumerate(self.patch_specs):
            threshold = tau[index] / mu.clamp_min(0.05)
            anchors.append(overlapping_patch_svt(value, threshold, patch_size, stride))
        anchor_stack = torch.stack(anchors, dim=1)
        robust_center = torch.median(anchor_stack, dim=1).values
        centered = value - value.mean(dim=(2, 3), keepdim=True)
        sigma = torch.sqrt(centered.square().mean(dim=(2, 3), keepdim=True) + EPS_SCALE**2)
        distance = torch.abs(anchor_stack - robust_center.unsqueeze(1)) / (sigma.unsqueeze(1) + EPS_SCALE)
        kappa = F.softplus(self.raw_kappa) + EPS_SCALE
        logits = self.branch_bias.view(1, 3, 1, 1, 1) - kappa * distance
        weights = torch.softmax(logits, dim=1)
        fused = (weights * anchor_stack).sum(dim=1)
        disagreement = torch.sqrt(
            (weights * (anchor_stack - fused.unsqueeze(1)).square()).sum(dim=1) + EPS_SCALE**2
        ) / (sigma + EPS_SCALE)
        return fused, anchor_stack, weights, disagreement


class GlobalAnchor(nn.Module):
    """Whole-image SVT ablation with the same output contract as patch consensus."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_tau = nn.Parameter(torch.tensor(_inverse_softplus(0.035)))

    def forward(
        self, value: torch.Tensor, mu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        threshold = (F.softplus(self.raw_tau) + EPS_SCALE) / mu.clamp_min(0.05)
        anchor = global_svt(value, threshold)
        stack = anchor.unsqueeze(1)
        weights = torch.ones_like(stack)
        disagreement = torch.zeros_like(anchor)
        return anchor, stack, weights, disagreement


class ReliabilityCalibratedWeight(nn.Module):
    """Stable log-domain fusion of structure and reciprocal sparse evidence."""

    def __init__(self, fusion: str = "reliability", blob_rescue: bool = False) -> None:
        super().__init__()
        if fusion not in {"reliability", "product"}:
            raise ValueError("fusion must be 'reliability' or 'product'")
        self.fusion = fusion
        self.blob_rescue = bool(blob_rescue)
        self.structure = FixedStructureTensor()
        self.raw_structure_slope = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.structure_bias = nn.Parameter(torch.tensor(0.0))
        self.raw_sparse_epsilon = nn.Parameter(torch.tensor(_inverse_softplus(0.01)))
        self.reliability_intercept = nn.Parameter(torch.tensor(0.5))
        self.raw_uncertainty_slope = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_blob_gamma = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        background: torch.Tensor,
        sparse: torch.Tensor,
        uncertainty: torch.Tensor,
        sparse_source: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coherence, energy, _ = self.structure(background)
        structure_slope = F.softplus(self.raw_structure_slope) + EPS_SCALE
        raw_structure = torch.log1p(
            F.softplus(structure_slope * coherence * energy + self.structure_bias)
        )
        q_background = _spatial_standardize(raw_structure)

        sparse_epsilon = F.softplus(self.raw_sparse_epsilon) + EPS_SCALE
        raw_sparse = -0.5 * torch.log(sparse.square() + sparse_epsilon.square())
        q_sparse = _spatial_standardize(raw_sparse)

        uncertainty_slope = F.softplus(self.raw_uncertainty_slope) + EPS_SCALE
        reliability = torch.sigmoid(self.reliability_intercept - uncertainty_slope * uncertainty)
        if self.fusion == "product":
            log_weight = q_background + q_sparse
        else:
            log_weight = reliability * q_background + (1.0 - reliability) * q_sparse

        blob_response = torch.zeros_like(background)
        blob_gamma = torch.zeros((), dtype=background.dtype, device=background.device)
        if self.blob_rescue:
            blob_response = self.structure.positive_blob_response(sparse_source)
            blob_gamma = 0.50 * torch.sigmoid(self.raw_blob_gamma)
            log_weight = log_weight - blob_gamma * _spatial_standardize(blob_response)

        # Subtracting the spatial maximum is algebraically neutral after mean
        # normalization and prevents float32 overflow for extreme evidence.
        stable_log_weight = log_weight - log_weight.amax(dim=(2, 3), keepdim=True)
        unnormalized = torch.exp(stable_log_weight)
        weight = unnormalized / (unnormalized.mean(dim=(2, 3), keepdim=True) + EPS_SCALE)
        weight = weight.clamp(0.25, 4.0)
        return {
            "weight": weight,
            "q_background": q_background,
            "q_sparse": q_sparse,
            "coherence": coherence,
            "energy": energy,
            "reliability": reliability,
            "blob_response": blob_response,
            "blob_gamma": blob_gamma,
        }


class RCPUnfoldStage(nn.Module):
    """One reliability-calibrated, preconditioned ADMM-inspired stage."""

    def __init__(
        self,
        hidden_channels: int = 24,
        patch_specs: Sequence[Tuple[int, int]] = ((8, 4), (12, 6), (16, 8)),
        fusion: str = "reliability",
        use_feedback: bool = True,
        blob_rescue: bool = False,
        anchor_mode: str = "patch",
        signed_sparse: bool = False,
    ) -> None:
        super().__init__()
        if anchor_mode not in {"patch", "global"}:
            raise ValueError("anchor_mode must be 'patch' or 'global'")
        self.use_feedback = bool(use_feedback)
        self.signed_sparse = bool(signed_sparse)
        self.anchor_mode = anchor_mode
        self.background_anchor = (
            PatchConsensusAnchor(patch_specs) if anchor_mode == "patch" else GlobalAnchor()
        )
        self.background_correction = BoundedCorrection(3, hidden_channels)
        self.weight_model = ReliabilityCalibratedWeight(fusion=fusion, blob_rescue=blob_rescue)
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

    def forward(
        self,
        image: torch.Tensor,
        background: torch.Tensor,
        sparse: torch.Tensor,
        dual: torch.Tensor,
        mu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        z_background = image - sparse + dual
        background_bar, patch_anchors, branch_weights, uncertainty = self.background_anchor(
            z_background, mu
        )
        background_scale = _spatial_rms(z_background).detach()
        background_delta, gamma_background = self.background_correction(
            background_scale, z_background, background_bar, uncertainty
        )
        background_next = background_bar + background_delta

        z_sparse = image - background_next + dual
        weight_trace = self.weight_model(background_next, sparse, uncertainty, z_sparse)
        sparse_weight = weight_trace["weight"]
        sparse_threshold = (F.softplus(self.raw_lambda) + EPS_SCALE) * sparse_weight / mu.clamp_min(0.05)
        threshold_fn = signed_soft_threshold if self.signed_sparse else positive_soft_threshold
        sparse_bar = threshold_fn(z_sparse, sparse_threshold)
        sparse_scale = _spatial_rms(z_sparse).detach()
        sparse_delta, gamma_sparse = self.sparse_correction(
            sparse_scale, z_sparse, sparse_bar, sparse_weight, uncertainty
        )
        corrected_sparse = sparse_bar + sparse_delta
        sparse_next = corrected_sparse if self.signed_sparse else F.relu(corrected_sparse)

        primal_residual = image - background_next - sparse_next
        eps_residual = 1.0e-8 * math.sqrt(float(image.shape[-2] * image.shape[-1]))
        primal_norm = torch.linalg.vector_norm(primal_residual, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        dual_norm = mu * torch.linalg.vector_norm(sparse_next - sparse, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        if self.use_feedback:
            alpha = 0.05 + 0.95 * torch.sigmoid(self.dual_gate(primal_residual))
            dual_star = dual + alpha * primal_residual
            eta = 0.25 * torch.sigmoid(self.raw_eta)
            balance = torch.tanh(torch.log((primal_norm + eps_residual) / (dual_norm + eps_residual)))
            mu_next = (mu * torch.exp(eta * balance)).clamp(0.05, 20.0)
            dual_next = (mu / mu_next) * dual_star
        else:
            alpha = torch.zeros_like(primal_residual)
            mu_next = mu
            dual_next = dual

        correction_bound_background = gamma_background * background_scale
        correction_bound_sparse = gamma_sparse * sparse_scale
        trace: Dict[str, torch.Tensor] = {
            "background": background_next,
            "sparse_intensity": sparse_next,
            "background_bar": background_bar,
            "sparse_bar": sparse_bar,
            "patch_anchors": patch_anchors,
            "branch_weights": branch_weights,
            "uncertainty": uncertainty,
            "w_rcp": sparse_weight,
            "q_background": weight_trace["q_background"],
            "q_sparse": weight_trace["q_sparse"],
            "coherence": weight_trace["coherence"],
            "structure_reliability": weight_trace["reliability"],
            "blob_response": weight_trace["blob_response"],
            "blob_gamma": weight_trace["blob_gamma"],
            "sparse_threshold": sparse_threshold,
            "primal_residual": primal_residual,
            "primal_norm": primal_norm,
            "dual_norm": dual_norm,
            "mu": mu,
            "mu_next": mu_next,
            "alpha": alpha,
            "background_delta": background_delta,
            "sparse_delta": sparse_next - sparse_bar,
            "background_correction_bound": correction_bound_background,
            "sparse_correction_bound": correction_bound_sparse,
            "background_scale": background_scale,
            "sparse_scale": sparse_scale,
        }
        return background_next, sparse_next, dual_next, mu_next, trace


class RCPRiRUFold(nn.Module):
    """Multi-stage RCP-RiRUFold network with physically separated outputs."""

    def __init__(
        self,
        stage_num: int = 5,
        hidden_channels: int = 24,
        patch_specs: Sequence[Tuple[int, int]] = ((8, 4), (12, 6), (16, 8)),
        fusion: str = "reliability",
        use_feedback: bool = True,
        blob_rescue: bool = False,
        anchor_mode: str = "patch",
        signed_sparse: bool = False,
    ) -> None:
        super().__init__()
        if stage_num < 1:
            raise ValueError("stage_num must be positive")
        self.stage_num = int(stage_num)
        self.stages = nn.ModuleList(
            [
                RCPUnfoldStage(
                    hidden_channels=hidden_channels,
                    patch_specs=patch_specs,
                    fusion=fusion,
                    use_feedback=use_feedback,
                    blob_rescue=blob_rescue,
                    anchor_mode=anchor_mode,
                    signed_sparse=signed_sparse,
                )
                for _ in range(self.stage_num)
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
            raise ValueError("RCPRiRUFold expects normalized Bx1xHxW input")

        background = image.clone()
        sparse = torch.zeros_like(image)
        dual = torch.zeros_like(image)
        centered = image - image.mean(dim=(2, 3), keepdim=True)
        mu = (5.0 * torch.sqrt(centered.square().mean(dim=(2, 3), keepdim=True) + EPS_SCALE**2)).clamp(
            0.05, 20.0
        )
        stage_traces: List[Dict[str, torch.Tensor]] = []
        prox_terms: List[torch.Tensor] = []

        for stage in self.stages:
            background, sparse, dual, mu, trace = stage(image, background, sparse, dual, mu)
            background_ratio = trace["background_delta"].abs().mean(dim=(1, 2, 3)) / (
                trace["background_scale"].reshape(-1) + EPS_SCALE
            )
            sparse_ratio = trace["sparse_delta"].abs().mean(dim=(1, 2, 3)) / (
                trace["sparse_scale"].reshape(-1) + EPS_SCALE
            )
            prox_terms.append((background_ratio + sparse_ratio).mean())
            stage_traces.append(trace)

        input_centered = image - image.mean(dim=(2, 3), keepdim=True)
        logit_scale = torch.sqrt(
            input_centered.square().mean(dim=(2, 3), keepdim=True) + EPS_SCALE**2
        ).clamp_min(MIN_LOGIT_SCALE).detach()
        logit_gain = F.softplus(self.raw_logit_gain) + EPS_SCALE
        logit_bias = F.softplus(self.raw_logit_bias) + EPS_SCALE
        logits = logit_gain * (sparse / logit_scale - logit_bias)

        if return_trace:
            detached_trace = [
                {name: value.detach() for name, value in stage.items()} for stage in stage_traces
            ]
            return background, logits, detached_trace
        if return_aux:
            aux = {
                "sparse_intensity": sparse,
                "prox_loss": torch.stack(prox_terms).mean(),
                "final_mu": mu,
                "final_dual": dual,
            }
            return background, logits, aux
        return background, logits


def build_rcp_model(name: str = "rirufold_rcp", **kwargs) -> RCPRiRUFold:
    """Build the main model or a preregistered ablation by server-facing name."""
    configurations = {
        "rirufold_rcp": {},
        "rirufold_rcp_product": {"fusion": "product"},
        "rirufold_rcp_nofeedback": {"use_feedback": False},
        "rirufold_rcp_blob": {"blob_rescue": True},
        "rirufold_rcp_global": {"anchor_mode": "global"},
        "rirufold_rcp_signed": {"signed_sparse": True},
    }
    if name not in configurations:
        raise ValueError("Unknown RCP-RiRUFold variant: {}".format(name))
    selected = dict(configurations[name])
    selected.update(kwargs)
    return RCPRiRUFold(**selected)
