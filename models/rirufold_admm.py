"""Proximal-anchored ADMM unfolding for infrared small-target detection.

This module intentionally lives beside the legacy ``RiRFold`` implementation.
It keeps explicit background, sparse, mask, dual, and penalty states so that
the network can be diagnosed as an unfolding model rather than only as a stack
of convolutional blocks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["RiRFoldADMM"]


def soft_threshold(value: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Element-wise proximal map of the weighted l1 norm."""
    return torch.sign(value) * F.relu(torch.abs(value) - threshold)


def singular_value_thresholding(value: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Batched SVT for Bx1xHxW feature maps.

    The compact SVD is deliberately retained as the low-rank proximal anchor.
    This is computationally heavier than a learned-only block, so it should be
    included in the runtime table when reporting this model.
    """
    matrix = value[:, 0]
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    shrunk = F.relu(singular - threshold.reshape(-1, 1))
    restored = (u * shrunk.unsqueeze(-2)) @ vh
    return restored.unsqueeze(1)


class ResidualCorrection(nn.Module):
    """Small learned correction around an explicit proximal anchor."""

    def __init__(self, in_channels: int, hidden_channels: int = 24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        # A small initial gate prevents the correction from immediately
        # overwhelming the mathematical proximal update.
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        correction = self.body(torch.cat(inputs, dim=1))
        gate = 0.25 * torch.sigmoid(self.gate_logit)
        return gate * correction, gate


class StructureTensorWeight(nn.Module):
    """Differentiable background-conditioned local-structure weight.

    High local coherence together with high gradient energy raises the sparse
    penalty, suppressing coherent background edges from entering the sparse
    target component. A small monotone calibration is learned, while the
    structure-tensor construction itself remains explicit.
    """

    def __init__(self, smooth_kernel: int = 5, min_weight: float = 0.35, max_weight: float = 3.0):
        super().__init__()
        self.smooth_kernel = smooth_kernel
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.log_slope = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, background: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x = F.conv2d(background, self.sobel_x, padding=1)
        grad_y = F.conv2d(background, self.sobel_y, padding=1)
        padding = self.smooth_kernel // 2
        jxx = F.avg_pool2d(grad_x.square(), self.smooth_kernel, stride=1, padding=padding)
        jyy = F.avg_pool2d(grad_y.square(), self.smooth_kernel, stride=1, padding=padding)
        jxy = F.avg_pool2d(grad_x * grad_y, self.smooth_kernel, stride=1, padding=padding)

        trace = jxx + jyy
        determinant = (jxx * jyy - jxy.square()).clamp_min(0.0)
        root = torch.sqrt((trace.square() - 4.0 * determinant).clamp_min(0.0) + 1e-12)
        lambda_1 = 0.5 * (trace + root)
        lambda_2 = 0.5 * (trace - root)
        coherence = (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + 1e-6)

        # Energy prevents flat, numerically coherent regions from receiving a
        # large structure penalty. The detached denominator is a scale normalizer.
        energy = trace / (trace.detach().mean(dim=(2, 3), keepdim=True) + 1e-6)
        structure_score = coherence * energy
        slope = F.softplus(self.log_slope) + 1e-4
        weight = 1.0 + F.softplus(slope * structure_score + self.bias)
        return weight.clamp(self.min_weight, self.max_weight), coherence


class SparseEvidenceWeight(nn.Module):
    """Stage-wise reciprocal sparse reweighting with bounded normalization."""

    def __init__(self, min_weight: float = 0.35, max_weight: float = 3.0):
        super().__init__()
        self.log_epsilon = nn.Parameter(torch.tensor(-4.6))
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(self, sparse: torch.Tensor) -> torch.Tensor:
        epsilon = F.softplus(self.log_epsilon) + 1e-6
        weight = torch.rsqrt(sparse.square() + epsilon)
        weight = weight / (weight.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return weight.clamp(self.min_weight, self.max_weight)


class CoupledWLSE(nn.Module):
    """Build W_LS(B), W_SE(S), and normalized W_LSE = W_LS * W_SE."""

    def __init__(self):
        super().__init__()
        self.structure = StructureTensorWeight()
        self.sparse = SparseEvidenceWeight()

    def forward(
        self,
        structure_source: torch.Tensor,
        sparse: torch.Tensor,
        use_structure: bool = True,
        dynamic_sparse: bool = True,
    ):
        if use_structure:
            w_ls, coherence = self.structure(structure_source)
        else:
            w_ls = torch.ones_like(structure_source)
            coherence = torch.zeros_like(structure_source)
        w_se = self.sparse(sparse) if dynamic_sparse else torch.ones_like(sparse)
        w_lse = w_ls * w_se
        w_lse = w_lse / (w_lse.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return w_ls, w_se, w_lse.clamp(0.25, 4.0), coherence


class ADMMUnfoldStage(nn.Module):
    """One proximal-anchored stage with a constrained dual preconditioner."""

    def __init__(
        self,
        hidden_channels: int = 24,
        use_svt: bool = True,
        use_feedback: bool = True,
        structure_source: str = "background",
        dynamic_sparse_weight: bool = True,
    ):
        super().__init__()
        self.use_svt = use_svt
        self.use_feedback = use_feedback
        self.structure_source = structure_source
        self.dynamic_sparse_weight = dynamic_sparse_weight
        self.log_tau = nn.Parameter(torch.tensor(-3.0))
        self.log_lambda = nn.Parameter(torch.tensor(-3.0))
        self.background_correction = ResidualCorrection(2, hidden_channels)
        self.sparse_correction = ResidualCorrection(3, hidden_channels)
        self.wlse = CoupledWLSE()
        self.dual_gate = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 3, padding=1),
        )
        self.dual_bias = nn.Parameter(torch.tensor(0.0))
        self.log_eta = nn.Parameter(torch.tensor(-2.2))

    def forward(
        self,
        image: torch.Tensor,
        background: torch.Tensor,
        sparse: torch.Tensor,
        dual: torch.Tensor,
        mu: torch.Tensor,
    ):
        z_background = image - sparse + dual
        tau = F.softplus(self.log_tau) / mu.clamp_min(1e-4)
        if self.use_svt:
            background_anchor = singular_value_thresholding(z_background, tau)
        else:
            # Kept only for the learned-only ablation; this is not SVT.
            background_anchor = F.avg_pool2d(z_background, 5, stride=1, padding=2)
        background_delta, gamma_background = self.background_correction(z_background, background_anchor)
        background_next = background_anchor + background_delta

        if self.structure_source == "background":
            structure_input = background_next
        elif self.structure_source == "input":
            structure_input = image
        elif self.structure_source == "none":
            structure_input = background_next
        else:
            raise ValueError(f"Unsupported structure source: {self.structure_source}")
        w_ls, w_se, w_lse, coherence = self.wlse(
            structure_input,
            sparse,
            use_structure=self.structure_source != "none",
            dynamic_sparse=self.dynamic_sparse_weight,
        )
        z_sparse = image - background_next + dual
        sparse_threshold = F.softplus(self.log_lambda) * w_lse / mu.clamp_min(1e-4)
        sparse_anchor = soft_threshold(z_sparse, sparse_threshold)
        sparse_delta, gamma_sparse = self.sparse_correction(z_sparse, sparse_anchor, w_lse)
        sparse_next = sparse_anchor + sparse_delta

        primal_residual = image - background_next - sparse_next
        if self.use_feedback:
            alpha = 0.05 + 0.95 * torch.sigmoid(self.dual_gate(primal_residual) + self.dual_bias)
            dual_next = dual + alpha * primal_residual
        else:
            alpha = torch.zeros_like(primal_residual)
            dual_next = dual

        # Residual-balanced, bounded penalty update. The use of batch-level
        # norms deliberately keeps mu scalar and interpretable.
        primal_norm = torch.linalg.vector_norm(primal_residual, dim=(1, 2, 3)).mean()
        dual_norm = mu * torch.linalg.vector_norm(sparse_next - sparse, dim=(1, 2, 3)).mean()
        if self.use_feedback:
            eta = 0.25 * torch.sigmoid(self.log_eta)
            balance = torch.tanh(torch.log((primal_norm + 1e-6) / (dual_norm + 1e-6)))
            mu_next = (mu * torch.exp(eta * balance)).clamp(0.05, 20.0)
        else:
            mu_next = mu

        trace = {
            "background": background_next,
            "sparse": sparse_next,
            "w_ls": w_ls,
            "w_se": w_se,
            "w_lse": w_lse,
            "coherence": coherence,
            "primal_residual": primal_residual,
            "primal_norm": primal_norm,
            "dual_norm": dual_norm,
            "mu": mu,
            "mu_next": mu_next,
            "alpha": alpha,
            "background_anchor": background_anchor,
            "sparse_anchor": sparse_anchor,
            "background_gate": gamma_background,
            "sparse_gate": gamma_sparse,
        }
        return background_next, sparse_next, dual_next, mu_next, trace


class RiRFoldADMM(nn.Module):
    """RiRUFold variant with explicit proximal anchors and co-evolving WLSE.

    By default ``forward(x)`` returns ``(background, sparse_logits)`` so the
    existing trainer can use it. ``return_trace=True`` exposes detached
    stage-wise states for diagnostics on a checkpoint.
    """

    def __init__(
        self,
        stage_num: int = 5,
        hidden_channels: int = 24,
        use_svt: bool = True,
        use_feedback: bool = True,
        structure_source: str = "background",
        dynamic_sparse_weight: bool = True,
        mode: str = "train",
    ):
        super().__init__()
        self.stage_num = stage_num
        self.mode = mode
        self.stages = nn.ModuleList(
            [
                ADMMUnfoldStage(
                    hidden_channels=hidden_channels,
                    use_svt=use_svt,
                    use_feedback=use_feedback,
                    structure_source=structure_source,
                    dynamic_sparse_weight=dynamic_sparse_weight,
                )
                for _ in range(stage_num)
            ]
        )

    def forward(self, image: torch.Tensor, return_trace: bool = False):
        background = image.clone()
        sparse = torch.zeros_like(image)
        dual = torch.zeros_like(image)
        # Use one scalar per batch, initialized from the observed image scale.
        mu = (5.0 * image.detach().std()).clamp(0.05, 20.0)
        trace: List[Dict[str, torch.Tensor]] = []

        for stage in self.stages:
            background, sparse, dual, mu, stage_trace = stage(image, background, sparse, dual, mu)
            if return_trace:
                trace.append({name: value.detach() for name, value in stage_trace.items()})

        if return_trace:
            return background, sparse, trace
        if self.mode == "train":
            return background, sparse
        return sparse
