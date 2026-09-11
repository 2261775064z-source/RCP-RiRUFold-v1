# RiRUFold: Interpretable ADMM-Unfolding Redesign Plan

> Evidence supplement: see `RiRUFold_ADMM_Evidence_Based_Design.md` for the
> attached TKDE manuscript's transferable viewpoints, the code-to-reviewer
> evidence chain, and a falsifiable validation plan for every proposed change.

## 1. Purpose and Scope

This document specifies a technically defensible redesign of RiRUFold. The goal
is not to add another generic attention or convolution block. The goal is to
make every stage retain a traceable relation to an ADMM update while introducing
one focused mechanism for infrared small-target detection: a background-sparse
co-evolving, structure-constrained reweighted mask.

The design has two levels:

1. **Implementation-consistency fixes**: required before making stronger
   interpretability claims in the manuscript.
2. **Model extension**: a background-conditioned WLSE and proximal-anchored
   learned correction. These are proposed contributions and must be trained and
   evaluated before being described as validated results.

The current plan is deliberately limited to single-frame ISTD. It does not claim
that the present three low-rank branches are literal tensor modes. A genuine
nonlocal patch-tensor formulation is listed only as future work.

---

## 2. Current Implementation Gaps

The following observations are based on `models/RiRFold.py` and `train.py`.
They should be fixed or explicitly reflected in the paper before new claims are
added.

### 2.1 Background output is not supervised

`RiRFold.forward()` returns the unchanged input `D` as its first output. The
training code applies `MSE(out_D, data)`, which is therefore identically zero.
The estimated low-rank backgrounds `B1`, `B2`, and `B3` are not returned to the
loss.

**Required correction**: return the final averaged background

\[
\mathbf{B}^K=\frac{1}{3}\sum_{j=1}^{3}\mathbf{B}_j^K
\]

and replace the zero-valued reconstruction term with a decomposition-consistency
loss and a mask-aware background loss.

### 2.2 The current sparse update is not an explicit weighted sparse proximal

The mathematical SRMT interpretation requires the reweighted mask to set the
sparse penalty. In the present `SparseModule`, `W*T` is processed by a dynamic
convolution and then added to the target estimate. Thus, the mask is not directly
used as the threshold of a weighted \(\ell_1\) proximal update.

**Required correction**: make weighted shrinkage the explicit main path and use
a learned term only as a bounded correction.

### 2.3 The penalty parameter is not stage-wise updated

The current `mu = 5 * torch.std(D)` is calculated once and reused by every
stage. If the manuscript displays a \(\mu^{k+1}\) update, the implementation and
the equation disagree.

**Required correction**: either remove the claimed penalty update from the
paper, or implement the bounded residual-balanced update in Section 5.4.

### 2.4 The structural weight is not yet an explicit structural-tensor prior

The current local branch consumes the image and gradient magnitude through a
CNN. This may be a useful feature extractor, but it cannot be described as a
strict structure-tensor eigenvalue construction unless that calculation is
implemented.

**Required correction**: use a differentiable structure-tensor descriptor as
the main structural cue; permit only a small learnable calibration on top of it.

---

## 3. Proposed Scientific Question

> How can a low-rank background estimate and an evolving sparse target estimate
> jointly determine a stage-wise sparse regularizer, while the resulting network
> remains anchored to explicit low-rank, sparse, mask, and dual ADMM updates?

This question differs from simply asking whether reweighting improves a score.
The proposed answer is that local structure should be estimated from the
progressively purified background, whereas target evidence should be estimated
from the current sparse component. The two sources are coupled only inside the
sparse regularization update.

---

## 4. Variables and Stage States

For an observed infrared image \(\mathbf{X}\), stage \(k\) maintains:

| Symbol | Meaning | Source |
|---|---|---|
| \(\mathbf{B}^k\) | low-rank/background estimate | low-rank update |
| \(\mathbf{S}^k\) | sparse target estimate | weighted sparse update |
| \(\mathbf{U}^k\) | scaled dual variable | decomposition residual |
| \(\mu_k\) | ADMM penalty parameter | residual-balanced schedule |
| \(\mathcal{W}_{\mathrm{LS}}^k\) | local-structure penalty map | \(\mathbf{B}^k\) |
| \(\mathcal{W}_{\mathrm{SE}}^k\) | sparse-evidence penalty map | \(\mathbf{S}^k\) |
| \(\mathcal{W}_{\mathrm{LSE}}^k\) | coupled reweighted sparse mask | product of both maps |

Use the **scaled-dual convention** throughout the code and manuscript. It avoids
ambiguous factors of \(\mu\) in the dual update.

---

## 5. Proposed Stage-Wise Update

### 5.1 Low-rank proximal anchor

Define the background input

\[
\mathbf{Z}_{B}^{k}=\mathbf{X}-\mathbf{S}^{k}+\mathbf{U}^{k}.
\]

The explicit anchor is

\[
\overline{\mathbf{B}}^{k+1}
= \operatorname{SVT}_{\tau_k}(\mathbf{Z}_{B}^{k}).
\]

The learnable background update is a bounded correction rather than an
unconstrained replacement:

\[
\mathbf{B}^{k+1}
= \overline{\mathbf{B}}^{k+1}
  + \gamma_B^k\, g_{\theta_k}
    (\mathbf{Z}_{B}^{k},\overline{\mathbf{B}}^{k+1}),
\qquad 0\leq\gamma_B^k\leq\gamma_{B,\max}.
\]

`LR_G` therefore has a precise role: it approximates or corrects an explicit
singular-value-thresholding update when the background is not perfectly low-rank.

### 5.2 Background-conditioned local-structure map

Compute image derivatives from the current background rather than directly from
the target-contaminated observation:

\[
\mathbf{J}^{k+1}
=G_\sigma *
\begin{bmatrix}
(B_x^{k+1})^2 & B_x^{k+1}B_y^{k+1}\\
B_x^{k+1}B_y^{k+1} & (B_y^{k+1})^2
\end{bmatrix}.
\]

Let \(\lambda_1\geq\lambda_2\) be eigenvalues of \(\mathbf J^{k+1}\). A local
coherence descriptor is

\[
c^{k+1}=\frac{\lambda_1-\lambda_2}
{\lambda_1+\lambda_2+\epsilon}.
\]

The structural penalty map is

\[
\mathcal{W}_{\mathrm{LS}}^{k+1}
=\operatorname{clip}\!\left(
\psi_{\omega}(c^{k+1}),w_{\min},w_{\max}\right),
\]

where \(\psi_\omega\) is a monotone, low-parameter calibration. It should map
coherent background structures to a stronger sparse penalty than weak isolated
target-like responses, according to the final chosen convention. The direction
of the map must be documented and verified numerically.

### 5.3 Sparse-evidence map and coupled WLSE

Use the current sparse response to form the classical reweighting component:

\[
\mathcal{W}_{\mathrm{SE}}^{k+1}
=\operatorname{clip}\!\left(
\frac{1}{\sqrt{(\mathbf{S}^{k})^2+\epsilon_k}},
w_{\min},w_{\max}\right).
\]

The proposed co-evolving mask is

\[
\mathcal{W}_{\mathrm{LSE}}^{k+1}
=\operatorname{normalize}\!\left(
\mathcal{W}_{\mathrm{LS}}^{k+1}
\odot\mathcal{W}_{\mathrm{SE}}^{k+1}\right).
\]

This is the central contribution. \(\mathcal{W}_{\mathrm{LS}}\) is refined
from the estimated background and \(\mathcal{W}_{\mathrm{SE}}\) is refined
from the estimated target. Therefore, it is not a one-pass multiplication of
two maps extracted from the original image.

### 5.4 Weighted sparse proximal anchor

Define

\[
\mathbf{Z}_{S}^{k}
=\mathbf{X}-\mathbf{B}^{k+1}+\mathbf{U}^{k}.
\]

The explicit weighted sparse update is

\[
\overline{\mathbf S}^{k+1}
=\operatorname{Soft}\!\left(
\mathbf{Z}_{S}^{k},
\frac{\lambda_k\mathcal W_{\mathrm{LSE}}^{k+1}}{\mu_k}
\right).
\]

The learned sparse correction is bounded:

\[
\mathbf S^{k+1}
=\overline{\mathbf S}^{k+1}
+\gamma_S^k h_{\phi_k}
(\mathbf Z_S^k,\overline{\mathbf S}^{k+1},
\mathcal W_{\mathrm{LSE}}^{k+1}).
\]

`SR_G` is thus a data-adaptive correction to a weighted sparse proximal step,
not an arbitrary target-segmentation CNN.

### 5.5 Residual feedback and penalty update

Use the explicit primal residual

\[
\mathbf R^{k+1}
=\mathbf X-\mathbf B^{k+1}-\mathbf S^{k+1}.
\]

The scaled dual update is

\[
\mathbf U^{k+1}
=\mathbf U^k+\alpha_k\odot\mathbf R^{k+1},
\qquad 0<\alpha_k\leq1.
\]

Here \(\alpha_k\) is a constrained learned preconditioner. It must be called a
preconditioned dual update in the paper, not an exact classical ADMM update.

Define scalar residual summaries

\[
r_k=\|\mathbf R^{k+1}\|_F,
\qquad
d_k=\mu_k\|\mathbf S^{k+1}-\mathbf S^k\|_F.
\]

Optionally update the penalty with a bounded residual-balancing schedule:

\[
\mu_{k+1}=\operatorname{clip}\!\left(
\mu_k\exp\left[
\eta_k\tanh\!\left(
\log\frac{r_k+\epsilon}{d_k+\epsilon}
\right)\right],\mu_{\min},\mu_{\max}\right).
\]

If this update is not implemented, delete every penalty-factor update claim
from the manuscript and figure captions.

---

## 6. Training Objective

Let \(\mathbf M\) be the ground-truth target mask. Use:

\[
\mathcal L
=\mathcal L_{\mathrm{seg}}
+\beta_{\mathrm{dc}}\mathcal L_{\mathrm{dc}}
+\beta_{\mathrm{bg}}\mathcal L_{\mathrm{bg}}
+\beta_{\mathrm{prox}}\mathcal L_{\mathrm{prox}}.
\]

Recommended terms:

\[
\mathcal L_{\mathrm{seg}}
=\operatorname{SoftIoU}(\mathbf S^K,\mathbf M)
+\beta_{\mathrm{bce}}\operatorname{BCEWithLogits}(\mathbf S^K,\mathbf M),
\]

\[
\mathcal L_{\mathrm{dc}}
=\|\mathbf X-\mathbf B^K-\mathbf S^K\|_1,
\]

\[
\mathcal L_{\mathrm{bg}}
=\|(1-\mathbf M)\odot(\mathbf B^K-\mathbf X)\|_1,
\]

\[
\mathcal L_{\mathrm{prox}}
=\frac{1}{K}\sum_{k=1}^{K}
\left(
\|\mathbf B^k-\overline{\mathbf B}^{k}\|_1
+\|\mathbf S^k-\overline{\mathbf S}^{k}\|_1
\right).
\]

`L_prox` should have a small coefficient or be introduced after the
segmentation loss has stabilized. It anchors learning to the proposed solver;
it should not force the learned model to reproduce an inadequate hand-crafted
proximal operator exactly.

---

## 7. Code-Level Refactor Plan

### Phase A: Correctness and trace export

Files to modify:

| File | Change |
|---|---|
| `models/RiRFold.py` | Return final background, target, and optional per-stage trace. |
| `models/RiRFold.py` | Replace unconstrained mask use with an explicit weighted soft-threshold main path. |
| `train.py` | Replace the zero-valued `MSE(out_D, data)` term with `L_dc` and `L_bg`. |
| `utils/` | Add a trace helper for primal residual, dual residual, weights, and penalty values. |
| `scripts/` | Add a held-out diagnostic exporter for stages 1...K. |

Suggested model interface:

```python
background, target, trace = net(image, return_trace=True)

# trace[k] contains:
# background, sparse, w_ls, w_se, w_lse,
# primal_residual, dual_residual, mu, alpha
```

### Phase B: Co-evolving WLSE

1. Compute `B_avg = (B1 + B2 + B3) / 3` after the low-rank update.
2. Generate the structure map from `B_avg`, not from the raw input.
3. Generate the sparse-evidence map from the previous/current sparse estimate.
4. Normalize and clamp both maps before multiplication.
5. Use the coupled map only in the sparse threshold.

### Phase C: Proximal-anchored corrections

1. Implement differentiable SVT for a small diagnostic model or an offline
   teacher target.
2. Implement exact weighted soft-thresholding in every stage.
3. Add bounded correction branches and `L_prox`.
4. Compare exact proximal, learned-only, and proximal-anchored variants.

### Phase D: Optional multi-scale consistency extension

This phase is optional. Assign distinct receptive fields or dilation rates to
the three background branches, and define their disagreement as

\[
\mathcal U_B^k=\operatorname{Var}(\mathbf B_1^k,\mathbf B_2^k,\mathbf B_3^k).
\]

Use \(\mathcal U_B^k\) only as a reliability calibration for the structural
weight. Do not present three same-architecture branches as literal tensor modes.

---

## 8. Required Experimental Evidence

No proposed design should enter the paper as a completed result before the
following experiments are run using real trained weights.

### 8.1 Architecture and operator ablations

| Variant | Explicit SVT | Explicit weighted shrinkage | Background-conditioned WLS | Dynamic WSE | Dual feedback | Purpose |
|---|---:|---:|---:|---:|---:|---|
| Fixed ADMM | yes | yes | fixed | yes | yes | classical reference |
| Learned-only | no | no | no | no | optional | black-box control |
| Proximal-anchored | yes | yes | no | no | yes | test proximal anchoring |
| Static WLSE | yes | yes | raw-input WLS | fixed | yes | test static prior |
| Co-evolving WLSE | yes | yes | background WLS | yes | yes | proposed mechanism |
| Full model | yes | yes | background WLS | yes | learned/preconditioned | full design |

Report mIoU, F1, Pd, Fa, parameters, FLOPs, inference time, final primal
residual, and final dual residual.

### 8.2 Conditioned test subsets

Partition the NUDT-SIRST test set before evaluation by:

1. local background structural complexity;
2. target SCR;
3. target area.

For each subset, report Pd and Fa in addition to mIoU/F1. The expected and
testable claim is narrow: co-evolving WLSE should mainly reduce structure-induced
false alarms and preserve weak targets in high-complexity, low-SCR scenes.

### 8.3 Diagnostic figures

For representative real images, show stage 1, middle stage, and final stage:

1. \(\mathbf B^k\), \(\mathbf S^k\), and \(|\mathbf X-\mathbf B^k-\mathbf S^k|\);
2. \(\mathcal W_{\mathrm{LS}}^k\), \(\mathcal W_{\mathrm{SE}}^k\), and
   \(\mathcal W_{\mathrm{LSE}}^k\);
3. per-stage primal and dual residual curves averaged over the real test set;
4. target-versus-background mean penalty weight curves.

The figures must come from trained checkpoints, not from a residual-proxy demo.

---

## 9. Manuscript and Rebuttal Positioning

### 9.1 Claims that are defensible after Phase B/C

Use wording such as:

> The contribution is not reciprocal sparse reweighting in isolation. The
> proposed mask couples a background-derived local structural prior with
> stage-wise sparse evidence inside the weighted sparse update. The low-rank,
> sparse, mask, and residual-feedback states are retained across unfolding
> stages.

For learned blocks:

> The learned branches are proximal-anchored corrections. They preserve the
> update inputs, output variables, and ordering of the SRMT solver, while
> adapting the fixed operators to heterogeneous infrared backgrounds.

### 9.2 Claims to avoid

Do not claim any of the following without a formal proof and matching code:

1. a new general reweighting principle;
2. global convergence of the learned unfolding network;
3. exact equivalence between convolution and a proximal operator;
4. stage-wise penalty updates when `mu` is constant;
5. literal tensor-mode updates when the implementation uses only parallel 2-D
   branches.

---

## 10. Recommended Execution Order

1. Refactor outputs and losses; verify that `L_dc` is nonzero and decreases.
2. Add trace export; run one existing checkpoint to inspect state values.
3. Implement background-conditioned explicit WLSE and weighted shrinkage.
4. Train baseline, static-WLSE, and co-evolving-WLSE under one fixed protocol.
5. Add proximal anchoring only if the explicit update baseline is stable.
6. Produce real diagnostic figures and conditional subgroup evaluations.
7. Update the manuscript and rebuttal only with the final trained results.

The first three items are the minimum technically sound path for a revision.
The nonlocal patch-tensor extension should be reserved for a new paper or a
substantial subsequent version.
