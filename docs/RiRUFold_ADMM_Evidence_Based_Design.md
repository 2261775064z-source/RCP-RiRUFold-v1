# RiRUFold ADMM-Unfolding: Evidence-Based Redesign Proposal

## Document Purpose

This is an evidence-based design proposal for the next RiRUFold iteration. It
extends `RiRUFold_ADMM_Design_Plan.md` by answering a stricter question for each
proposed change:

> What observation motivates the change, which argument from the attached TKDE
> manuscript supports the reasoning, which reviewer concern does it answer, and
> what real experiment could disprove it?

The document does **not** treat a design idea as an experimental conclusion.
Only results obtained from real trained checkpoints may enter the manuscript or
the rebuttal as empirical evidence.

---

## 1. How to Read the Attached TKDE Manuscript

### 1.1 Its actual contribution

The attached manuscript is titled *Spatially Coupled Low-Rank Tensor Completion
for Zero-Observation Spatio-Temporal Data Reconstruction* (called **SCLT**
below). It is an interpretable ADMM tensor-reconstruction method for spatial and
temporal data. It is **not a deep unfolding network**: its low-rank factor and
reconstruction updates are not replaced by learned neural modules.

Consequently:

* Do not cite it as a deep-unfolding network baseline.
* Do not borrow its zero-observation or coordinate-interpolation setting for a
  single-frame ISTD paper.
* Do use it as an example of how to defend an ADMM-based tensor model: separate
  information channels, define their optimization roles, derive every update,
  isolate components experimentally, and state a bounded conclusion.

The attached PDF is marked as an initial-submission proof. Before citing it in a
paper, verify final publication metadata, author list, venue, year, volume, and
DOI.

### 1.2 Important viewpoints extracted from SCLT

The following are faithful **paraphrases** of the source article, with article
section references. They are included as arguments to learn from, rather than
text to copy.

| Evidence | Important viewpoint from SCLT | Why the viewpoint matters |
|---|---|---|
| E1 | Sec. I and III-A distinguish two reconstruction targets because each has a different information source. | A prior is meaningful only if it supplies information that another component cannot provide. |
| E2 | Sec. III-C defines the spatial prior as a standalone operator, with a clear input, output, and stability property. | A module should be described as an operator with an identifiable role, not merely a feature block. |
| E3 | Sec. III-D inserts the spatial term and low-rank term into one objective, then identifies what is lost when either term is removed. | The novelty of coupling is established through the objective and ablation, not by placing modules adjacent in a figure. |
| E4 | Sec. III-E derives the update sequence before presenting the algorithm. Low-rank, reconstruction, and dual variables each have an explicit update. | An unfolding architecture should retain the solver's state variables, inputs, outputs, and order. |
| E5 | The paragraph after Algorithm 1 identifies repeated exchange within an iterative loop as the difference from a one-pass composition. | A product of two weights becomes a coupled mechanism only if both are updated and re-used across iterations. |
| E6 | Sec. III-E computes primal and dual residuals; Algorithm 1 uses them to assess iteration progress. | An ADMM interpretation needs residual diagnostics, even if no global convergence theorem is claimed. |
| E7 | Sec. IV organizes experiments as research questions, separating information sufficiency, component complementarity, and operating conditions. | Aggregate mIoU alone cannot verify a proposed mechanism. |
| E8 | Sec. V reports that the coupled model has an operating regime and may lose advantage in some boundary conditions. | A technically credible paper states when its added prior helps and when it may not. |
| E9 | Sec. V-D includes sensitivity and imperfect-mask analysis. | A reweighted-mask method needs robustness evidence for its weights and hyperparameters. |

### 1.3 Correct analogy and incorrect analogy

| SCLT concept | Valid RiRUFold interpretation | Interpretation to avoid |
|---|---|---|
| Spatial prior | A local structural prior derived from the estimated infrared background. | Geographic IDW interpolation between image pixels. |
| Low-rank repair | Low-rank background separation. | CP completion over unavailable sensor/time dimensions. |
| Coupled ADMM loop | Background, target, mask, and residual states repeatedly condition one another. | A one-time `WLS * WSE` multiplication. |
| Boundary experiments | Structural complexity, SCR, target-size, and mask-reliability subsets. | Zero-observation-node experiments. |

---

## 2. RiRUFold's Current Technical Position

The existing source is `models/RiRFold.py`; training is in `train.py`. The
following facts need to be recognized before extending the method.

| Current implementation fact | Consequence for the paper |
|---|---|
| `RiRFold.forward()` returns unchanged `D` and `T`; `D` is the input image. | `MSE(out_D, data)` in `train.py` is zero. The current loss does not supervise background decomposition. |
| `LowrankModule` is a residual CNN operating on three parallel branches. | It is a learnable background estimator, but not an explicit SVT update and not automatically a tensor-mode update. |
| `SparseModule` processes `W*T` with a dynamic convolution. | The code does not yet make `W_LSE` the threshold of a weighted sparse proximal map. |
| `WeightUpdateNetwork` takes raw input `F` and sparse target `T`. | The structural weight is influenced by the unseparated image; it is not background-conditioned or strictly static/analytic. |
| `mu` is initialized once from `torch.std(D)` and is never updated. | A manuscript equation claiming `mu^{k+1}` is inconsistent with the code. |
| `AdaptiveStepGate` predicts a pixelwise residual multiplier. | It can be called a learned dual preconditioner only after constraints and residual evidence are added; it is not an exact ADMM dual update. |

These are not merely implementation details. They directly relate to the two
most serious reviewer concerns: (i) WLSE appears to be a simple combination of
known weights, and (ii) the unfolding modules appear to be black-box fitting.

---

## 3. Core Redesign Thesis

### 3.1 Proposed scientific statement

The proposed contribution should be formulated as:

> RiRUFold uses a background-sparse co-evolving reweighted mask. At each stage,
> the local structural penalty is estimated from the currently recovered
> background, the sparse-evidence penalty is estimated from the current target
> component, and their coupled map explicitly controls the next weighted sparse
> proximal update.

This is narrower and more defensible than claiming a new generic reweighting
principle.

### 3.2 State loop

For image `X`, preserve the following stage state:

\[
(B^k,S^k,U^k,\mu_k,W_{\mathrm{LS}}^k,W_{\mathrm{SE}}^k,
W_{\mathrm{LSE}}^k).
\]

The intended information loop is:

\[
B^k \longrightarrow W_{\mathrm{LS}}^{k+1},\qquad
S^k \longrightarrow W_{\mathrm{SE}}^{k+1},
\]

\[
(W_{\mathrm{LS}}^{k+1},W_{\mathrm{SE}}^{k+1})
\longrightarrow W_{\mathrm{LSE}}^{k+1}
\longrightarrow S^{k+1},
\]

\[
(B^{k+1},S^{k+1}) \longrightarrow
R^{k+1}=X-B^{k+1}-S^{k+1}
\longrightarrow U^{k+1}.
\]

This loop is the relevant lesson from E5. It makes the contribution a
stage-dependent coupling mechanism, not a static combination of `WLS` and
`WSE`.

---

## 4. Proposal P0: Repair the Decomposition Training Contract

### Problem

The current loss has no effective background term because the model returns the
input as `out_D`. Therefore, the claimed decomposition `X = B + S` is not
directly trained or measured.

### Evidence chain

* **SCLT E3**: every information channel is placed in an explicit objective.
* **SCLT E4/E6**: reconstructed variables and residuals are retained by the
  solver and checked during iteration.
* **Reviewer C2**: black-box modules cannot be called unfolding updates if the
  decomposition constraint is neither optimized nor measured.

### Design

Return the final background estimate and target estimate:

```python
background, target, trace = net(image, return_trace=True)
```

Use a decomposition-consistency loss:

\[
L_{\mathrm{dc}}=\|X-B^K-S^K\|_1.
\]

Use a background loss outside the annotated target:

\[
L_{\mathrm{bg}}=\|(1-M)\odot(B^K-X)\|_1.
\]

The full objective becomes

\[
L=L_{\mathrm{seg}}+\beta_{\mathrm{dc}}L_{\mathrm{dc}}
+\beta_{\mathrm{bg}}L_{\mathrm{bg}}.
\]

### Code changes

* `models/RiRFold.py`: return final `B_avg` rather than unchanged `D`.
* `models/RiRFold.py`: retain `B_avg` in every stage trace.
* `train.py`: delete `MSE(out_D, data)` and compute `L_dc` plus `L_bg`.

### Required evidence

Report mean final `L_dc` on train and test, plus stage-wise
`||X-B^k-S^k||_F / ||X||_F`. The final residual should be lower than for the
no-feedback version. If not, do not use decomposition consistency as an argument
for the dual branch.

---

## 5. Proposal P1: Background-Sparse Co-evolving WLSE

### Problem

The existing structural path uses the raw image. It has no explicit mechanism
for becoming less target-contaminated as decomposition improves. Consequently,
the present `W_LSE = W_local * W_sparse` can reasonably appear to be a simple
combination of existing weighting techniques.

### Evidence chain

* **SCLT E1**: different information sources should retain separate roles.
* **SCLT E2**: a prior should have a clear operator input and output.
* **SCLT E5**: repeated exchange, not one-pass composition, establishes the
  meaning of coupling.
* **Reviewer C1**: the present paper must demonstrate more than a combination
  of popular adaptive weights.

### Design

After low-rank estimation, form

\[
B_{\mathrm{avg}}^{k+1}=\frac{B_1^{k+1}+B_2^{k+1}+B_3^{k+1}}{3}.
\]

Compute the structural and sparse maps from different states:

\[
W_{\mathrm{LS}}^{k+1}=f_{\mathrm{struct}}(B_{\mathrm{avg}}^{k+1}),
\]

\[
W_{\mathrm{SE}}^{k+1}=
\operatorname{clip}\left(
\frac{1}{\sqrt{(S^k)^2+\epsilon_k}},w_{\min},w_{\max}\right),
\]

\[
W_{\mathrm{LSE}}^{k+1}=
\operatorname{normalize}
\left(W_{\mathrm{LS}}^{k+1}\odot W_{\mathrm{SE}}^{k+1}\right).
\]

`W_LSE` must control the sparse threshold in Proposal P3. It should not be fed
only as a generic feature multiplier.

### Code changes

Replace:

```python
W = self.weight(D, T)
```

with a stage-aware interface conceptually equivalent to:

```python
B_avg = (B1 + B2 + B3) / 3
W_ls, W_se, W_lse = self.weight(B_avg, T, previous_weight=W)
```

The exact update order must be fixed in the manuscript and code. Do not use a
caption that says `W_LSE` is updated before `B` if the code computes it after
`B`.

### Required evidence

Train four real variants under the same seed, stages, losses, and split:

| Variant | WLS source | WSE | Purpose |
|---|---|---|---|
| Point-WSE | none | dynamic | sparse weighting alone |
| Static WLSE | raw input | fixed/dynamic | one-pass or static coupling |
| Background WLS + fixed WSE | recovered background | fixed | isolate background conditioning |
| Co-evolving WLSE | recovered background | dynamic | proposed mechanism |

Report global metrics and metrics on high-structure / low-SCR subsets. The key
acceptance signal is lower false alarm or higher Pd in the intended difficult
subset, not only a minor overall mIoU change.

### Claim boundary

If the full model only improves a proxy experiment, write that it illustrates a
mechanism; do not claim it proves the trained network's advantage. If the full
model improves only high-structure scenes, state exactly that rather than
claiming universal superiority.

---

## 6. Proposal P2: Explicit Structure-Tensor Prior with Small Learnable Calibration

### Problem

The current `local_struct_net` is a CNN supplied with image intensity and a
gradient magnitude. It cannot support a strict claim that the local mask is
derived from structure-tensor eigenvalues.

### Evidence chain

* **SCLT E2**: the spatial operator is explicit, transparent, stable, and has
  identifiable input/output semantics.
* **SCLT E9**: an auxiliary mask/prior must be tested for robustness.
* **Reviewer C1/C2**: WSE and associated modules are criticized as existing
  weighting plus black-box fitting.

### Design

Obtain image derivatives from `B_avg`, form the local structure tensor

\[
J=G_\sigma *
\begin{bmatrix}
B_x^2 & B_xB_y\\
B_xB_y & B_y^2
\end{bmatrix},
\]

and compute its coherence

\[
c=\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\epsilon}.
\]

Define `WLS = psi(c)` using a monotone function with a few learnable scalars,
for example slope, bias, and clipping bounds. The manuscript must specify the
weight direction: high weight means a high weighted-`l1` sparse penalty.

### Required evidence

Compare gradient-only weighting, free CNN weighting, and explicit
structure-tensor-plus-calibration. For each, report:

1. target-region mean weight;
2. background-region mean weight;
3. target/background penalty ratio;
4. mIoU, F1, Pd, and Fa;
5. sensitivity to `epsilon`, smoothing scale, and clipping limits.

### Claim boundary

If the explicit structure map does not improve or stabilize the method, call the
branch a learned structural estimator. Do not retain a structure-tensor claim
that is unsupported by code and ablation.

---

## 7. Proposal P3: Proximal-Anchored Learned Corrections

### Problem

`LR_G` and `SR_G` are currently residual/dynamic convolution modules. The
operator mapping table says they approximate low-rank and sparse proximal
operators, but there is no direct proximal anchor or numerical fidelity check.

### Evidence chain

* **SCLT E3/E4**: the objective and derived updates come before the algorithm.
* **SCLT E6**: iterative diagnostics show whether the solver interpretation is
  actually preserved.
* **Reviewer C2**: standard convolutions and residual blocks are not, by
  themselves, an explanation of why the architecture represents a proximal step.

### Design

Use an explicit proximal main path:

\[
\bar B^{k+1}=\operatorname{SVT}_{\tau_k}(X-S^k+U^k),
\]

\[
\bar S^{k+1}=
\operatorname{Soft}\left(
X-B^{k+1}+U^k,
\frac{\lambda_kW_{\mathrm{LSE}}^{k+1}}{\mu_k}
\right).
\]

Then permit bounded corrections:

\[
B^{k+1}=\bar B^{k+1}+\gamma_B^k g_{\theta_k}(\cdot),
\qquad 0\leq\gamma_B^k\leq\gamma_{B,\max},
\]

\[
S^{k+1}=\bar S^{k+1}+\gamma_S^k h_{\phi_k}(\cdot),
\qquad 0\leq\gamma_S^k\leq\gamma_{S,\max}.
\]

The network no longer claims that a convolution *is* a proximal operator. It
claims that a learned correction adapts an explicit proximal step to complex
infrared backgrounds.

### Required evidence

Compare fixed-proximal ADMM, learned-only updates, and proximal-anchored
updates. Report mIoU/F1/Pd/Fa, runtime, parameters, final residuals, correction
magnitudes, and mean proximal deviations:

\[
\|B^k-\bar B^k\|_1,
\qquad \|S^k-\bar S^k\|_1.
\]

### Risk

SVT can be expensive at 256x256. A valid practical compromise is periodic SVT
teacher supervision or a truncated-SVD implementation. It is not valid to call
the existing learned-only module proximal-anchored without an explicit or
supervisory proximal reference.

---

## 8. Proposal P4: Residual-Consistent Dual Feedback and Penalty Update

### Problem

The code uses an adaptive residual gate, but the paper's ADMM notation may imply
an exact dual update. In addition, `mu` is constant although figures/formulas
may show a stage update.

### Evidence chain

* **SCLT E4**: dual variables are a distinct state, not a generic residual
  feature.
* **SCLT E6**: primal and dual residuals are measured during iterations.
* **Reviewer C2**: the role of residual feedback must be theoretically and
  experimentally explained.

### Design

Let

\[
R^{k+1}=X-B^{k+1}-S^{k+1}.
\]

Use a constrained learned dual preconditioner:

\[
U^{k+1}=U^k+\alpha_k\odot R^{k+1},
\qquad 0<\alpha_k\leq 1.
\]

Call this a **preconditioned dual update**, not exact ADMM. Log the stage mean,
minimum, and maximum of `alpha_k`.

If a penalty update is implemented, use a bounded residual-balancing rule:

\[
r_k=\|R^{k+1}\|_F,
\qquad d_k=\mu_k\|S^{k+1}-S^k\|_F,
\]

\[
\mu_{k+1}=\operatorname{clip}\left(
\mu_k\exp\left[\eta_k\tanh
\left(\log\frac{r_k+\epsilon}{d_k+\epsilon}\right)\right],
\mu_{\min},\mu_{\max}\right).
\]

If this is not trained and tested, remove the `mu`-update formula from figures,
algorithm boxes, tables, and captions.

### Required evidence

Compare no feedback, fixed dual step, learned bounded dual preconditioner, and
the optional residual-balanced penalty schedule. Evidence for the dual role is a
lower or more stable real residual trajectory, not only an mIoU increase.

---

## 9. Proposal P5: Give the Three Background Branches a Defensible Role

### Problem

The current `B1`, `B2`, and `B3` branches have the same conceptual role. They
should not be presented as three tensor-mode updates for a single 2-D image.

### Evidence chain

* **SCLT E2/E3**: every channel has a specified information role and ablation.
* **Reviewer C2**: unexplained repeated residual branches are perceived as
  black-box fitting.

### Design

Use distinct receptive fields/dilations and state that branches estimate
small-, middle-, and large-scale backgrounds. Define an optional disagreement
map:

\[
U_B^k=\operatorname{Var}(B_1^k,B_2^k,B_3^k).
\]

Use `U_B` only to reduce confidence in `WLS` where background reconstruction is
inconsistent. This adds an interpretable reliability signal without claiming
that the branches are tensor modes.

### Required evidence

Compare same-receptive-field branches, explicit multi-scale branches, and
multi-scale branches with uncertainty calibration. This is optional and should
not be pursued if P1-P4 already consume the revision budget.

---

## 10. Proposal P6: Conditioned Evaluation, Sensitivity, and Boundary Claims

### Problem

The main benchmark table cannot verify that WLSE solves structure-induced false
alarms or weak-target loss. It only proves an aggregate ranking.

### Evidence chain

* **SCLT E7**: component experiments must answer separate questions.
* **SCLT E8**: method benefits should be stated conditionally.
* **SCLT E9**: mask-based methods need sensitivity and reliability analysis.
* **Reviewer C1/C3**: novelty and completeness require more than one global
  ablation and more than a single added baseline.

### Experimental protocol

Before comparing methods, partition the NUDT-SIRST test set by:

1. local structure complexity, using a pre-defined background gradient or
   structure-tensor score;
2. target SCR;
3. target area;
4. optionally, target-to-strong-edge distance.

For each subset report mIoU, F1, Pd, and Fa. Run sensitivity sweeps over
`epsilon`, weight clipping, `lambda`, stage count, and structure-map perturbation.
The grouping rule must be fixed before viewing model results.

### Required conclusion format

Use a bounded conclusion such as:

> Co-evolving WLSE provides its clearest benefit in scenes with strong local
> structure and weak sparse target evidence; the advantage is reduced when the
> background is already homogeneous or the target is sufficiently salient.

Only use this text if results support it. A negative result is still useful: it
defines the true boundary of the proposed prior.

---

## 11. Revision Artifact Map

| Planned artifact | Required content | Proposal(s) | Reviewer concern |
|---|---|---|---|
| SRMT objective and algorithm | Variables, states, map insertion point, update order, scaled-dual convention | P0-P4 | C1, C2, C5 |
| Fig. 12 | `B -> WLS`, `S -> WSE`, `WLSE -> weighted shrinkage`, residual -> dual feedback | P1, P4 | C1, C2, C5 |
| Operator mapping table | Exact anchor, learnable correction, input/output state, residual diagnostic | P3, P4 | C2 |
| WLSE ablation table | Static/dynamic and coupled/uncoupled real trained variants | P1, P2 | C1 |
| Stage trace figure | Real `B`, `S`, `WLS`, `WSE`, `WLSE`, primal/dual residuals from trained weights | P0-P4 | C1, C2 |
| Subgroup table | High/low structure, low/high SCR, target size | P1, P6 | C1, C3 |
| Sensitivity figure | Epsilon, lambda, clipping, stage count, structure-map noise | P2, P6 | C1, C3 |
| Runtime table | Fixed ADMM, learned-only, proximal-anchored, full RiRUFold | P3 | C2, C3 |

The existing residual-proxy WLSE demo is supplementary mechanism intuition only.
It must not be presented as the primary proof for a trained RiRUFold model.

---

## 12. Priority and Decision Gate

### Minimum revision path

1. Implement P0 and trace export.
2. Implement P1 with explicit weighted sparse shrinkage.
3. Run static versus co-evolving WLSE with real training.
4. Add P4 residual diagnostics and the no-feedback comparison.
5. Add P6 subgroup and sensitivity analysis.

This path directly addresses C1 and C2 without rebuilding the entire paper.

### Stronger but more expensive path

Add P2 and P3, with explicit structure-tensor construction and
proximal-anchored learned corrections. This gives the strongest interpretability
story, but it requires careful implementation and retraining.

### Do not do in the current revision unless results are already available

Do not redesign the model around a true nonlocal patch tensor or claim literal
tensor-mode unfolding. That is a valid next-paper direction, but it changes the
architecture, training cost, and experimental baseline too substantially for a
defensible short revision.
