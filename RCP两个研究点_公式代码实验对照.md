# RCP-RiRUFold 两个研究点：公式、代码与实验对照

> 本文档是当前 RCP 方案的权威说明。旧文档中“patch-SVT 锚点”“多个局部低秩观点”“鲁棒共识”等表述不再使用。正式论文、代码注释、实验名称和结果图均以本文档为准。

## 1. 方法定位：两篇论文如何融合

RCP 不是把两篇论文的模块直接拼接，而是按“问题骨架—建模原则—新模型”融合：

| 来源 | 保留的核心 | 在 RCP 中如何变化 |
|---|---|---|
| `docs/manuscript_zb.pdf` | 单帧红外图像的低秩背景 `B`、稀疏目标 `S`、重加权稀疏阈值和深度展开状态 | 保留 `B/S/U/W` 的物理含义；重新设计背景低秩更新和稀疏权重更新 |
| `docs/TKDE-2026-08-2894_Proof_hi (1).pdf` | 多信息源必须进入显式目标或显式更新；交替过程要有可观测状态、边界和反证实验 | 把“不同尺度的背景估计”和“背景结构/稀疏状态两类证据”写成可追踪张量，并为每个研究点设置独立消融 |
| 本研究 | 两篇论文均未直接给出的机制 | 研究点一：中位偏差加权的多尺度块提升低秩背景估计；研究点二：尺度分歧门控的双证据稀疏重加权 |

论文主张只保留两个研究点。硬有界校正、逐图像 `mu` 更新、对偶缩放和 SVD 稳定梯度属于实现保障，不计作研究点。

## 2. 研究点一：多尺度块提升低秩背景估计与中位偏差融合

### 2.1 要解决的模型问题

整图矩阵 SVT 对整幅图使用同一种空间尺度。云层、平滑天空、建筑边缘和目标邻域的结构尺度不同，单一整图矩阵可能把非平稳背景混在同一低秩关系中。仅增加三个分支后做平均也不充分，因为某个尺度明显偏离另外两个尺度时，平均仍给它固定的 `1/3` 权重。

因此研究点一同时包含两项不可缺少的模型变化：

1. 从一个整图矩阵改成三个尺度条件的块提升矩阵；
2. 从固定平均改成无尺度偏置的中位偏差加权融合。

### 2.2 三个尺度在公式中的明确含义

第 `k` 阶段的背景输入为

$$
Z_B^k=X-S^k+U^k.
$$

尺度集合固定为

$$
(p_s,h_s)\in\{(8,4),(12,6),(16,8)\},
$$

其中 `p_s` 是块大小，`h_s` 是步长。令 $R_{s,n}$ 提取第 `n` 个 $p_s\times p_s$ 图像块，则

$$
\mathcal P_s(Z_B^k)=
[\operatorname{vec}(R_{s,1}Z_B^k),\ldots,
\operatorname{vec}(R_{s,N_s}Z_B^k)]
\in\mathbb R^{p_s^2\times N_s}.
$$

三种矩阵的行数分别是 `64`、`144`、`256`。需要特别强调：当前代码不是对每个小图块各做一次 SVD，而是先把同一尺度的所有图块放到同一个矩阵中，再对该矩阵做一次 SVD。以 `256×256` 输入为例，经过当前边界填充后，三个尺度对应的矩阵约为：

| 尺度 `(p_s,h_s)` | 块提升矩阵大小 | 该尺度的 SVD 次数 |
|---|---:|---:|
| `(8,4)` | `64×4225` | 1 |
| `(12,6)` | `144×1849` | 1 |
| `(16,8)` | `256×1089` | 1 |

矩阵的每一列是一个图块；SVT 利用不同图块之间的共同结构压缩奇异值。三个尺度最终产生三个与输入同尺寸的背景图 `B_8、B_12、B_16`。它们不是三个张量模态，也不是三个互不相干的区域分解；模型中唯一需要使用的名称就是“三个尺度背景估计”。

每个尺度的阈值为

$$
\theta_{k,s}=\frac{\operatorname{softplus}(\alpha_{k,s})+\epsilon}
{\max(\mu_k,0.05)}.
$$

对 $\mathcal P_s(Z_B^k)=Q_{k,s}\operatorname{diag}(\sigma_{k,s,i})V_{k,s}^{\top}$ 做

$$
\widehat{\mathcal P}_{k,s}=Q_{k,s}
\operatorname{diag}[(\sigma_{k,s,i}-\theta_{k,s})_+]V_{k,s}^{\top}.
$$

令 $D_s=\sum_nR_{s,n}^{\top}R_{s,n}$ 为重叠覆盖计数矩阵，尺度背景估计为

$$
B_s^{k+1}=D_s^{-1}\sum_nR_{s,n}^{\top}
\operatorname{mat}(\widehat{\mathcal P}_{k,s}[:,n]).
$$

这与代码中的 `unfold → SVT → fold → coverage normalization → crop` 完全对应。由于重叠块提取不是正交变换，论文只称其为“显式背景估计器”，不声称它是重叠核范数目标在图像域的精确近端算子。

### 2.3 逐像素尺度权重：降低什么、为什么降低、在模型哪里使用

被降低的是某个尺度背景值 $B_s(h,w)$ 进入最终背景 $\widetilde B(h,w)$ 时的融合系数 $\omega_s(h,w)$。它不是卷积核参数的权重，不是损失函数权重，也不改变 SVT 的奇异值阈值。

在同一个像素位置 $(h,w)$，三个尺度给出三个背景估计：

$$
[B_8(h,w),B_{12}(h,w),B_{16}(h,w)].
$$

逐像素中位数 $M_B(h,w)$ 只用于回答“哪个尺度的该像素结果偏离另外两个更多”，它不是最终背景。输入空间尺度 $\sigma_Z^k$ 只用于消除不同图像亮度范围的影响，使暗图和亮图中的偏差可以使用相同的 $\kappa_k$。二者定义为：

$$
M_B^{k+1}=\operatorname{median}_s B_s^{k+1},
$$

$$
\sigma_Z^k=\sqrt{\operatorname{mean}_{hw}
(Z_B^k-\overline Z_B^k)^2+\epsilon^2}.
$$

第 `s` 个尺度在该像素处的归一化偏差为

$$
d_s^{k+1}=\frac{|B_s^{k+1}-M_B^{k+1}|}{\sigma_Z^k+\epsilon}.
$$

主模型把该偏差变成逐像素融合权重

$$
\omega_s^{k+1}=
\frac{\exp(-\kappa_kd_s^{k+1})}
{\sum_{t=1}^{3}\exp(-\kappa_kd_t^{k+1})},
\qquad
\kappa_k=\operatorname{softplus}(\kappa_{k,raw})+\epsilon>0.
$$

代码已经删除旧版 `branch_bias`，因此不存在“偏差更大但因学习偏置而仍获得更高权重”的歧义。保持其他分支的偏差不变时

$$
\frac{\partial\omega_s}{\partial d_s}
=-\kappa_k\omega_s(1-\omega_s)<0.
$$

因此偏差越大的尺度，在该像素进入背景融合时所占比例越小。融合背景为

$$
\widetilde B^{k+1}=\sum_{s=1}^{3}\omega_s^{k+1}B_s^{k+1}.
$$

例如同一像素的三个背景估计为 `[0.20, 0.21, 0.60]`，中位数是 `0.21`。第三个尺度的偏差远大于前两个尺度，因此 `B_16=0.60` 进入该像素最终背景时的系数会被降低。反过来，如果小尺度过度跟随高频杂波而得到 `[0.55, 0.22, 0.21]`，被降低的是小尺度的系数。

这一步在 unfolding 中替换的是原来的单一背景更新：

```text
原背景路径：Z_B^k -> whole-image SVT -> B_bar^(k+1)

新背景路径：Z_B^k
          -> 三个 [patch lifting -> matrix SVT -> coverage-normalized fold]
          -> B_8、B_12、B_16
          -> d_s -> omega_s -> B_tilde^(k+1)
          -> bounded correction -> B^(k+1)
```

所以 `omega_s` 的作用位置在背景更新内部、`B^(k+1)` 形成之前。随后 `B^(k+1)` 才进入

$$
Z_S^k=X-B^{k+1}+U^k
$$

并影响稀疏目标更新。尺度分歧 `Delta_B` 还会进入研究点二，改变稀疏阈值；这构成从背景更新到稀疏更新的显式耦合。

归一化尺度分歧为

$$
\Delta_B^{k+1}=
\frac{\sqrt{\sum_s\omega_s^{k+1}
(B_s^{k+1}-\widetilde B^{k+1})^2+\epsilon^2}}
{\sigma_Z^k+\epsilon}.
$$

`Δ_B` 只能发现尺度间不一致；若三个尺度同时产生相同方向的误差，则 `Δ_B` 可能很小。这是需要在论文限制部分明确写出的共同误差边界。

### 2.4 研究点一的代码位置

| 公式/变量 | 代码对象 | 输出键 |
|---|---|---|
| $\mathcal P_s$、SVT、覆盖归一化 | `overlapping_patch_svt` | 每个 `B_s` |
| 三尺度估计 | `MultiScalePatchLowRankEstimator` | `scale_backgrounds` |
| $d_s\rightarrow\omega_s\rightarrow\widetilde B$ | `median_deviation_fusion` | `scale_weights`、`background_bar` |
| $\Delta_B$ | `median_deviation_fusion` | `scale_disagreement` |
| 整图对照 | `GlobalLowRankEstimator` | `rirufold_rcp_global` |
| 等权对照 | `fusion_mode="mean"` | `rirufold_rcp_mean` |

权威实现文件：`models/rirufold_rcp.py`。

### 2.5 研究点一必须完成的实验

| 对比 | 只改变什么 | 能回答的问题 |
|---|---|---|
| `rirufold_rcp` vs `rirufold_rcp_global` | 三个块提升矩阵 vs 一个整图矩阵 | 多尺度块提升背景估计是否必要 |
| `rirufold_rcp` vs `rirufold_rcp_mean` | 偏差权重 vs 固定 `1/3` | 中位偏差融合是否有效 |
| 单尺度 `8/12/16` 与两尺度组合 | 尺度集合 | 改进是否仅由某一个尺度贡献 |
| 同 FLOPs/参数量对照 | 计算预算 | 增益是否只是更多计算造成 |

研究点一成立的最低条件：完整模型在至少两个数据集上稳定优于 `global` 和 `mean`，且尺度权重图能显示高偏差尺度被降低，而不是所有尺度长期保持 `1/3`。

### 2.6 Transformer 与小型 SVT 的可选实现

这一方向可以研究，但必须区分三种不同方案：

1. **Transformer 预测 SVT 阈值。** Transformer 读取 patch token，只预测正阈值 $\theta_{s,g}$，显式 SVT 仍然保留。它属于 unfolding 阶段内的可学习参数生成器，低秩解释最完整。
2. **Transformer 分组 + 小型矩阵 SVT。** Transformer 或窗口注意力先把相关 patch 分成大小为 `K` 的组，再对每组 $p_s^2\times K$ 矩阵做 SVT。这比“对每个单独 patch 做 SVD”更合理，因为它仍利用多个 patch 之间的共同结构。
3. **Transformer 完全替代 SVT。** 此时模型不再执行奇异值阈值，论文只能称“学习型低秩代理”或“低秩瓶颈网络”，必须增加秩约束、核范数代理或因子分解约束，不能继续声称显式 SVT 更新。

推荐的支线是第 2 种。令 patch token 为

$$
t_{s,n}=E_s\operatorname{vec}(R_{s,n}Z_B^k)+e_{s,n}^{pos},
\qquad H_s=\operatorname{WindowTransformer}_s(\{t_{s,n}\}),
$$

由 $H_s$ 给出窗口内分组 $\mathcal G_{s,g}$ 和正阈值 $\theta_{s,g}$，然后

$$
\mathcal P_{s,g}(Z_B^k)=
[\operatorname{vec}(R_{s,n}Z_B^k)]_{n\in\mathcal G_{s,g}}
\in\mathbb R^{p_s^2\times K},
$$

$$
\widehat{\mathcal P}_{s,g}
=\operatorname{SVT}_{\theta_{s,g}}(\mathcal P_{s,g}).
$$

最后把所有组覆盖归一化回投为 $B_s$，再沿用当前的 $d_s\rightarrow\omega_s\rightarrow\widetilde B$。这样改变的是研究点一的低秩估计器，而研究点二可以保持不变，便于公平对比。

从工程结构看，这不是两个完整模型串联，而是一个 unfolding stage 内包含一个 Transformer 子模块和一个解析 SVT 算子。只有把 ViT 单独训练、再把它的输出送入另一个完整 RCP 网络时，才属于真正的级联“网络套网络”。全局 ViT 对数千个 patch 做 $N_s^2$ 注意力代价很高，建议使用窗口 Transformer、稀疏注意力或固定窗口分组。

## 3. 研究点二：尺度分歧门控的双证据稀疏重加权

### 3.1 要解决的模型问题

原始固定乘积把背景结构证据和当前稀疏状态证据始终同时施加到阈值上。当研究点一的三个背景估计明显不一致时，继续固定使用结构证据可能把边缘或云层误差传入稀疏阈值。研究点二让可观测的 `Δ_B` 决定结构证据在当前像素应占多少比例。

### 3.2 两类证据如何计算

由背景图的 Sobel 梯度和 `5×5` 二项式平滑核得到结构张量

$$
J_B=G*\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix},
\qquad \lambda_1\ge\lambda_2\ge0.
$$

结构一致性和归一化能量为

$$
c_B=\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\epsilon},
\qquad
e_B=\frac{\lambda_1+\lambda_2}
{\operatorname{mean}_{hw}(\lambda_1+\lambda_2)+\epsilon}.
$$

结构证据和稀疏状态证据为

$$
q_B=\mathcal N_{sp}\left(
\log[1+\operatorname{softplus}(a_cc_Be_B+b_c)]\right),
$$

$$
q_S=\mathcal N_{sp}\left(
-\frac12\log[(S^k)^2+\epsilon_{S,k}^2]\right).
$$

`N_sp` 对每张图的空间维单独标准化，不在 batch 之间共享统计量。

### 3.3 分歧如何改变稀疏阈值

定义正斜率

$$
a_{d,k}=\operatorname{softplus}(a_{d,k,raw})+\epsilon>0,
$$

结构证据门控为

$$
\pi_B^{k+1}=\sigma(a_{0,k}-a_{d,k}\Delta_B^{k+1}).
$$

它满足

$$
\frac{\partial\pi_B}{\partial\Delta_B}
=-a_{d,k}\pi_B(1-\pi_B)<0.
$$

所以 `Δ_B` 增大时，结构证据系数必然降低。双证据对数融合为

$$
h_W=\pi_Bq_B+(1-\pi_B)q_S.
$$

稳定化和均值归一化后

$$
W_{RCP}=\operatorname{clip}\left(
\frac{\exp(h_W-\max_{hw}h_W)}
{\operatorname{mean}_{hw}\exp(h_W-\max_{hw}h_W)+\epsilon},
0.25,4\right).
$$

该权重直接进入稀疏近端更新：

$$
Z_S^k=X-B^{k+1}+U^k,
$$

$$
\bar S^{k+1}=\max\left(
Z_S^k-\frac{\lambda_kW_{RCP}^{k+1}}{\max(\mu_k,0.05)},0\right).
$$

因此研究点二改变的是可解释的加权 `l1` 阈值，不是附加一个无法追踪的注意力特征。

### 3.4 研究点二的代码位置

| 公式/变量 | 代码对象 | 输出键 |
|---|---|---|
| $c_B,e_B$ | `FixedStructureTensor` | `coherence`、`energy` |
| $q_B,q_S$ | `DisagreementGatedSparseWeight` | `q_background`、`q_sparse` |
| $\Delta_B\rightarrow\pi_B$ | `DisagreementGatedSparseWeight` | `structure_gate` |
| $W_{RCP}$ | `DisagreementGatedSparseWeight` | `w_rcp` |
| 加权稀疏阈值 | `RCPUnfoldStage.forward` | `sparse_threshold` |
| 固定门控对照 | `mode="fixed_gate"` | `rirufold_rcp_fixedgate` |
| 固定乘积对照 | `mode="product"` | `rirufold_rcp_product` |

### 3.5 研究点二必须完成的实验

| 对比/干预 | 只改变什么 | 能回答的问题 |
|---|---|---|
| `rirufold_rcp` vs `rirufold_rcp_fixedgate` | 学习的空间门控 vs `π_B=0.5` | 分歧驱动门控是否必要 |
| `rirufold_rcp` vs `rirufold_rcp_product` | 凸组合对数池化 vs `q_B+q_S` | 是否优于原始固定乘积逻辑 |
| `π_B=0/0.5/1` | 结构证据比例 | 性能对门控方向是否符合公式 |
| 空间打乱 `π_B` | 保留分布、破坏位置 | 收益是否来自空间对应关系 |
| `Δ_B` 分箱 | 分歧与背景误差/Fa 的关系 | 能否进一步使用“可靠性”解释 |

研究点二成立的最低条件：完整模型稳定优于 `fixedgate` 与 `product`；正常空间门控优于打乱门控。只有 `Δ_B` 与背景误差/虚警存在稳定相关时，论文才能增加“结构可靠性校准”的解释，否则只写“尺度分歧门控”。

## 4. 支持模块：不能冒充研究点

| 支持模块 | 作用 | 论文定位 |
|---|---|---|
| `BoundedCorrection` | 将 CNN 修正限制在每图 RMS 的 `0.20` 内 | 实现稳定性与可追踪性 |
| 逐图像残差平衡 `mu` | 避免 batch 内图像相互改变罚参数 | 数值合同 |
| scaled-dual rescaling | `mu` 变化时保持未缩放对偶变量一致 | 状态一致性 |
| deterministic SVD surrogate | 重复奇异值附近保持有限梯度，同时恢复精确前向值 | 训练数值保障 |
| monotone readout | 将物理稀疏强度与分割 logits 分开 | 输出语义保障 |

## 5. 代码、公式和论文的一一对应

主文件：

- 网络：`RiRUFold_ADMM/models/rirufold_rcp.py`
- 数值烟测：`RiRUFold_ADMM/scripts/smoke_test_rirufold_rcp.py`
- 训练后状态导出：`RiRUFold_ADMM/scripts/diagnose_rirufold_rcp.py`
- 服务器评测：`RiRUFold_ADMM/scripts/evaluate_rirufold_rcp.py`
- 完整消融矩阵：`RiRUFold_ADMM/scripts/run_server_ablation.sh`
- 英文 LaTeX 方法：`paper_manuscript_variants/RCP_RiRUFold_Paper/latex/sections/02_method.tex`
- 英文 LaTeX 实验：`paper_manuscript_variants/RCP_RiRUFold_Paper/latex/sections/03_experiments.tex`

`return_trace=True` 的权威变量名：

| 研究阶段 | 变量名 |
|---|---|
| 三尺度背景结果 | `scale_backgrounds` |
| 三尺度偏差权重 | `scale_weights` |
| 归一化尺度分歧 | `scale_disagreement` |
| 结构证据门控 | `structure_gate` |
| 稀疏重加权 | `w_rcp` |
| 稀疏阈值 | `sparse_threshold` |

## 6. 本地测试结果

本地测试只验证公式和实现，不代表训练精度：

| 检查 | 当前结果 |
|---|---:|
| 零阈值 unfold/fold 最大绝对误差 | `1.1920929e-6` |
| 偏差 `0.5` 时的离群尺度权重 | `0.2326966` |
| 偏差增至 `1.5` 后的同一尺度权重 | `0.1003677` |
| batch 组成对背景结果的最大影响 | `0.0` |
| batch 组成对 logits 的最大影响 | `0.0` |
| 非有限梯度 | `0` |
| 等权消融 | 每尺度严格 `1/3` |
| 固定门控消融 | `π_B` 严格 `0.5` |

测试命令：

```bash
cd /data/Student-25/ZhangYao/RPCANet-main
python RiRUFold_ADMM/scripts/smoke_test_rirufold_rcp.py
python RiRUFold_ADMM/scripts/smoke_test_server_pipeline.py
python RiRUFold_ADMM/scripts/evaluate_rirufold_rcp.py --help
python RiRUFold_ADMM/scripts/diagnose_rirufold_rcp.py --help
```

## 7. 服务器训练与评测命令

一轮链路验证：

```bash
cd /data/Student-25/ZhangYao/RPCANet-main
GPU_ID=0 OUTPUT_ROOT=./result_rirufold_rcp_dryrun EPOCHS=1 SEEDS="42" \
bash RiRUFold_ADMM/scripts/run_server_ablation.sh
```

三数据集、三种子正式矩阵：

```bash
cd /data/Student-25/ZhangYao/RPCANet-main
GPU_ID=0 OUTPUT_ROOT=./result_rirufold_rcp_main EPOCHS=400 \
SEEDS="42 3407 2026" bash RiRUFold_ADMM/scripts/run_server_ablation.sh
```

默认顺序已经包含：

```text
rirufold_admm
rirufold_rcp_global
rirufold_rcp_mean
rirufold_rcp_product
rirufold_rcp_fixedgate
rirufold_rcp
rirufold_rcp_nofeedback
```

主结论必须使用同一数据划分、epoch、seed、batch size 和阈值协议。`latest.pkl` 作为固定末轮主结果；不能把测试集选出的 `best.pkl` 混入主表。

## 8. 正式论文必须补的图

| 图 | 必须展示的内容 | 对应研究点 |
|---|---|---|
| 总体网络图 | `P_s(Z_B) → B_s → d_s → ω_s → B_tilde, Δ_B → π_B → W_RCP → S` | 两个研究点 |
| 三尺度背景图 | 输入、三个 `B_s`、三个 `d_s/ω_s`、融合背景 | 研究点一 |
| 权重单调性图 | `d_s` 横轴、`ω_s` 纵轴，显示单调下降 | 研究点一 |
| RP1 消融图 | global、mean、full 的 mIoU/Fa/耗时 | 研究点一 |
| 分歧门控图 | `Δ_B`、`π_B`、`q_B`、`q_S`、`W_RCP`、阈值 | 研究点二 |
| 门控反事实图 | learned、0、0.5、1、shuffle 的配对变化 | 研究点二 |
| RP2 消融图 | product、fixedgate、full 的三数据集结果 | 研究点二 |
| 失败案例图 | 三尺度共同误差、强边缘、暗目标、扩展目标 | 限制与反证 |

## 9. 统一术语

| 不再使用 | 统一改为 |
|---|---|
| patch-SVT 锚点 | 尺度条件的块提升低秩背景估计 |
| 多个局部低秩观点 | 三个尺度条件的块提升矩阵/背景估计 |
| 鲁棒共识 | 无尺度偏置的中位偏差加权融合 |
| 离群分支 | 相对逐像素中位估计偏差较大的尺度 |
| 不确定性 | 归一化尺度分歧 `Δ_B` |
| 结构可靠性概率 | 结构证据门控 `π_B`；通过校准实验后才可讨论可靠性 |

一句话论文表述：

> RCP-RiRUFold 首先用三个尺度条件的块提升低秩估计替代单一整图背景更新，并以无偏置的中位偏差规则降低不一致尺度的贡献；随后将其归一化尺度分歧映射为结构证据门控，直接改变加权稀疏阈值。
