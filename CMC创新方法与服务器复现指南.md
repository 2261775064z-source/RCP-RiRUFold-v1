# CMC-RiRUFold 第二套创新方法、公式修改与服务器复现指南

## 1. 方案定位

第二套方案命名为 **Counterfactual Masked-Completion RiRUFold（CMC-RiRUFold）**。
它与第一套 RCP-RiRUFold 保留相同的“低秩背景—稀疏目标—逐阶段展开—物理稀疏量与
分割 logits 分离”框架，但研究重点不同：

- RCP 解决“稀疏阈值的结构证据是否可靠”；
- CMC 解决“疑似目标位置是否还应被当作可靠背景观测，以及缺失背景如何补全”。

CMC 只在推理时读取 `X/B/S`，内部候选掩膜 `C` 不读取标签，也不是第二个分割头。
本方案从 `manuscript_zb.pdf` 继承低秩—稀疏深度展开根方向，从
`TKDE-2026-08-2894_Proof_hi (1).pdf` 迁移“局部观测不足时引入受约束侧信息并交替更新”
的建模原则；红外侧信息来自同一幅图的可靠背景环域，不照搬地理 IDW。

已有红外研究包含 outlier/superpixel masking，因此本文不主张“首次使用掩膜”。可主张
的组合创新边界是：**阶段候选观测排除、行随机同图背景侧先验、两步掩蔽 patch 低秩
补全、正精度共识和候选条件稀疏近端被放进同一个可诊断展开闭环**。

## 2. 从基础公式到 CMC 的修改

### 2.1 基础低秩—稀疏分解

基础模型写成

\[
\min_{B,S}\ \|B\|_*+\lambda\|S\|_1,
\qquad X=B+S.
\]

普通背景更新默认所有像素都可直接约束 `B≈X`。红外小目标正位于高亮异常处，这会把
目标能量重新吸收到背景。CMC 首先改变观测合同，而不是再增加一个自由卷积模块。

### 2.2 无标签候选观测排除

固定 `9×9` 均值核为 `G_9`，定义

\[
H=\operatorname{ReLU}(X-G_9*X),\qquad
D^k=\operatorname{ReLU}(X-B^k),
\]

\[
\mathcal R(A)=\frac{A}{\sqrt{\operatorname{mean}_{hw}(A^2)+\epsilon_{score}^2}},
\]

\[
A^k=\zeta_H^k\mathcal R(H)+\zeta_S^k\mathcal R(S^k)
+\zeta_D^k\mathcal R(D^k),\qquad \boldsymbol\zeta^k=\operatorname{softmax}(v_k).
\]

候选面积从 2% 固定收缩到分辨率下界：

\[
\varrho_{min}=\max(1/HW,5\times10^{-4}),\quad
\varrho_k=0.02-(0.02-\varrho_{min})\frac{k}{\max(K-1,1)}.
\]

令 `q_k` 为 `A^k` 的停止梯度 `(1-varrho_k)` 分位数，使用总体标准差 `s_k`：

\[
g_k=\frac{s_k}{s_k+0.05},\qquad
t_k=[0.02+0.18\sigma(t_{raw,k})](s_k+\epsilon_{score}),
\]

\[
C^k=g_k\sigma((A^k-q_k)/t_k),\qquad V^k=1-C^k.
\]

`g_k` 保证平坦输入严格得到 `C=0`。训练只加无标签预算约束

\[
L_{budget}^k=[\operatorname{mean}_{hw}C^k-g_k\varrho_k]^2,
\]

禁止用 test mask 调候选面积。

### 2.3 行随机反事实背景侧先验

固定两个非负环形核：`K_5` 中心 `3×3` 为零、外围 16 点各为 `1/16`；`K_9`
中心 `5×5` 为零、外围 56 点各为 `1/56`。对任意当前状态 `Y`，

\[
d_m^k=K_m*V^k+\epsilon_A,
\]

\[
B_{A,m}(Y;V^k)=\frac{K_m*(V^k\odot Y)+\epsilon_A Y}{d_m^k},
\]

\[
\xi_m^k=\operatorname{softmax}_m(b_m+\log d_m^k),\qquad
\mathcal A_k(Y)=\sum_m\xi_m^kB_{A,m}(Y;V^k).
\]

给定 `V` 时，每个输出像素是非负、和为 1 的输入凸组合，因此
`\mathcal A_k(c)=c` 且 `||\mathcal A_k z||_infinity<=||z||_infinity`。这给侧信息一个
可检验的稳定性边界，而不是自由 attention。

### 2.4 条件掩蔽补全目标与两步展开

令 `Z_B^k=X-S^k+U^k`，条件目标为

\[
F_{(k,j)}(L)=\frac{p_{X,k}}2\|\sqrt{V^k}\odot(L-X)\|_F^2
+\frac{p_{L,k}}2\|L-Z_B^k\|_F^2
+\frac{p_{A,k}}2\|\sqrt{C^k}\odot(L-B_A^{(k,j)})\|_F^2
+\sum_{s=1}^3\tau_{k,s}\|P_s(L)\|_*.
\]

三个正精度为

\[
p_{r,k}=0.25+3.75\sigma(p_{raw,r,k}),\qquad r\in\{X,L,A\}.
\]

正常模型初始化 `(p_X,p_L,p_A)=(2,1,2)`：可靠区优先真实输入，候选区让反事实侧
信息强于可能仍含目标的 `Z_B`。首先

\[
B_A^{(k,0)}=\mathcal A_k(B^k),\qquad
L^{k,0}=V^k\odot X+C^k\odot B_A^{(k,0)}.
\]

每阶段固定执行两次 gradient-guided unfolded denoising：

\[
G_{k,j}=p_XV^k\odot(L^{k,j}-X)+p_L(L^{k,j}-Z_B^k)
+p_AC^k\odot(L^{k,j}-B_A^{(k,j)}),
\]

\[
\nu_k=(p_X+p_L+p_A+\epsilon_L)^{-1},
\]

\[
Q^{k,j}=L^{k,j}-\nu_kG_{k,j},\qquad
L^{k,j+1}=\operatorname{PCSVT}_{\nu_k\tau_k}(Q^{k,j}),
\]

\[
B_A^{(k,j+1)}=\mathcal A_k(L^{k,j+1}),\qquad j=0,1.
\]

最终共识为

\[
\bar B^{k+1}=\frac{p_XV^k\odot X+p_LL^{k,2}+p_AC^k\odot B_A^{(k,2)}}
{p_XV^k+p_L+p_AC^k}.
\]

分母因 `p_L>0.25` 严格为正，不添加会破坏权重和为 1 的额外 epsilon。重叠 patch
PCSVT 只称去噪锚点，不声称精确求解 `argmin F` 或具有单调下降证明。

### 2.5 候选条件稀疏近端

结构证据 `q_B` 与倒数稀疏证据 `q_S` 仍逐样本空间标准化，但不复用 RCP 的可靠性门：

\[
\log\widetilde W_{CMC}^{k+1}=q_B^{k+1}+q_S^k
-\beta_{C,k}(C^k-\operatorname{mean}_{hw}C^k),\quad 0<\beta_{C,k}<0.5,
\]

\[
W_{CMC}^{k+1}=\operatorname{clip}\left(
\frac{\exp(\log\widetilde W_{CMC}^{k+1})}
{\operatorname{mean}_{hw}\exp(\log\widetilde W_{CMC}^{k+1})},0.25,4\right),
\]

\[
\bar S^{k+1}=\max\left(X-B^{k+1}+U^k-
\frac{\lambda_kW_{CMC}^{k+1}}{\mu_k},0\right).
\]

背景与稀疏两侧再加 `0.20×local_RMS×tanh(net)` 硬有界校正，并复用第一方案已经
修正的逐样本 residual-balanced `mu` 与 scaled-dual 重标。

### 2.6 训练目标

\[
L_{CMC}=L_{softIoU}+\beta_{dc}L_{dc}+\beta_{bg}L_{bg}+\beta_{prox}L_{prox}
+\beta_{budget}\overline L_{budget}+\beta_{cf}L_{cf},
\]

\[
L_{cf}=\operatorname{mean}_k
\frac{\operatorname{mean}_{hw}(C^k\odot|B^{k+1}-B_A^{(k,J)}|)}
{\operatorname{mean}_{hw}(C^k)+\epsilon_C}.
\]

默认 `beta_budget=0.1`、`beta_cf=0.01`。`noside` 明确令 `p_A=0`、`L_cf=0`，并且
代码路径完全不调用 `\mathcal A_k`。

## 3. 网络结构与代码映射

```text
X, B^k, S^k, U^k, mu_k
        |
        v
CandidateObservationMask ----> C^k, V^k
        |
        v
RowStochasticAnnularPrior ----> B_A^(k,0)
        |
        v
2 x [gradient step -> 3-scale PatchConsensusSVT -> refresh side]
        |
        v
PositivePrecisionConsensus -> bounded B correction
        |
        v
CandidateConditionedWeight -> positive sparse prox -> bounded S correction
        |
        v
per-sample residual / mu update / scaled-dual rescale
```

| 数学模块 | 代码位置 |
|---|---|
| `C/V`、固定面积日程、平坦门 | `models/rirufold_cmc.py::CandidateObservationMask` |
| `K_5/K_9` 行随机侧先验 | `RowStochasticAnnularPrior` |
| `F_(k,j)` 的梯度步与两次侧信息刷新 | `CMCUnfoldStage.forward` |
| 三尺度 PCSVT | 复用 `rirufold_rcp.py::PatchConsensusAnchor` |
| 正精度闭式共识 | `CMCUnfoldStage.forward` 的 `denominator/numerator` |
| 候选条件阈值 | `CandidateConditionedWeight` |
| 多阶段、fixedmask 缓存、logits 读出 | `CMCRiRUFold` |
| 五个服务器模型名 | `build_cmc_model` |
| 训练新增损失 | `integration_reference/train.py::compute_loss` |

## 4. 消融和可否证结论

| 模型名 | 唯一改变 | 回答的问题 |
|---|---|---|
| `rirufold_cmc` | 完整 CMC | 主模型 |
| `rirufold_cmc_nomask` | 所有阶段 `C=0,V=1` | 改变背景观测是否必要 |
| `rirufold_cmc_noside` | 不调用侧先验，`p_A=0,L_cf=0` | 侧信息是否真正贡献 |
| `rirufold_cmc_onepass` | 每阶段只做一次 inner 更新 | 第二次交换是否必要 |
| `rirufold_cmc_fixedmask` | 只计算一次 `C^0/V^0`，并冻结其阶段 0 面积预算 | 阶段重估是否必要 |

若服务器上 `onepass` 不劣于完整模型，不得把“两步交换”单列为有效贡献；若
`noside` 与完整模型无差异，则侧先验主张不成立。CMC 相对第一方案 RCP 进入论文主表
的预注册门槛是：至少 2/3 数据集三种子平均 mIoU 提高不低于 0.30 个百分点，Fa
相对增加不超过 5%，且目标区背景吸收率下降。

## 5. 本地门禁与机制结果

```powershell
cd D:\python_DM\RPCANet-main\RiRUFold_ADMM
.\.venv\Scripts\python.exe scripts\smoke_test_rirufold_cmc.py
.\.venv\Scripts\python.exe scripts\smoke_test_cmc_server_pipeline.py
.\.venv\Scripts\python.exe scripts\run_cmc_research.py --seed 20260912
```

本地 smoke 已验证：常量保持误差 `1.19e-7`、平坦图 `C_max=0`、完整阶段侧先验调用
次数 3、两次 inner 改变量均非零、最小融合分母约 3、`noside` 调用次数 0、梯度有限、
batch 不变性通过、真实 NUDT 图 trace 通过。

8 个按 NUDT 中位目标占比校准的 40×40 合成场景得到以下**未训练机制中位数**：

| 配置 | 目标区背景吸收 MAE↓ | 目标-边缘稀疏对比↑ | 边缘稀疏响应↓ |
|---|---:|---:|---:|
| CMC | 0.03620 | 0.38280 | 0.000372 |
| No mask | 0.41787 | 0 | 0 |
| No side | 0.41556 | 0 | 0 |
| One pass | 0.03639 | 0.38357 | 0.000334 |
| Fixed mask | 0.04340 | 0.37301 | 0.001620 |

这些数字支持“掩蔽+侧先验能阻止合成目标被背景吸收”的机制方向，但 `onepass` 和完整
模型仍非常接近，因此多步优势必须由服务器训练决定。它们不是 mIoU/F1/Pd/Fa，不能
写入论文精度主表。原始数值在 `results/cmc_*.csv/json`，候选图在
`figures/*_q2_*`。

## 6. 服务器集成

在服务器原 `RPCANet-main` 中：

1. 把 `RiRUFold_ADMM/models/rirufold_cmc.py` 与 `rirufold_rcp.py` 复制到原工程
   `models/`；CMC 的 PCSVT 复用后者。
2. 审阅 `integration_reference/models___init__.py`，把 CMC import 和
   `name.startswith('rirufold_cmc')` 分支合入原 `models/__init__.py`。
3. 审阅 `integration_reference/train.py`，把 `return_aux`、物理稀疏重构损失、
   `--mask-budget-weight` 和 `--cf-weight` 合入原 `train.py`。
4. 先运行单元 smoke，再启动正式训练。不要直接覆盖服务器已有自定义修改。

### 6.1 单模型训练示例

```bash
cd /data/Student-25/ZhangYao/RPCANet-main
CUDA_VISIBLE_DEVICES=0 python train.py \
  --net-name rirufold_cmc --dataset nudt \
  --data-root ./RiRUFold_ADMM/datasets \
  --epochs 400 --lr 1e-4 --batch-size 4 --gpu 0 --seed 42 \
  --admm-stage-num 5 --admm-hidden-channels 24 \
  --dc-weight 0.05 --bg-weight 0.02 --prox-weight 0.005 \
  --mask-budget-weight 0.1 --cf-weight 0.01 \
  --save-iter-step 100 --log-per-iter 10 \
  --base-dir ./result_rirufold_cmc \
  --run-name seed_42/nudt/rirufold_cmc
```

IRSTD-1K 与 SIRST-Aug 只替换：

```bash
--dataset irstd1k --data-root ./datasets
--dataset sirstaug --data-root ./datasets
```

### 6.2 三数据集、三种子、全部消融

```bash
cd /data/Student-25/ZhangYao/RPCANet-main
bash RiRUFold_ADMM/scripts/run_cmc_server_ablation.sh
```

可覆盖环境变量，例如：

```bash
GPU_ID=1 EPOCHS=400 SEEDS="42 3407 2026" \
OUTPUT_ROOT=./result_rirufold_cmc_v1 \
bash RiRUFold_ADMM/scripts/run_cmc_server_ablation.sh
```

脚本检测到目标 run 目录已存在时会拒绝覆盖。

训练和独立评测全部完成后，脚本还会调用严格结果分析器。只有
`3 数据集 × 7 模型 × 3 种子=63` 个单元齐全、全部为完整测试、同数据集 split hash
与 threshold 一致，且每个实验 checkpoint SHA-256 合法、不被其他单元重复使用时才生成：

- `analysis/server_seed_records.csv`：逐种子原始记录；
- `analysis/server_summary.csv`：各数据集/模型 mean±std；
- `analysis/server_paired_gains.csv`：CMC 相对各对照的逐 seed 配对增益及 bootstrap CI；
- `analysis/result_q2_server_{main,ablation,paired_gain}.{png,svg}`：论文结果图；
- `analysis/CMC服务器结果报告.md`：可直接审阅的自动报告。

若需要在评测完成后单独重建这些证据：

```bash
python RiRUFold_ADMM/scripts/analyze_cmc_server_results.py \
  --input-root ./result_rirufold_cmc/metrics \
  --output-dir ./result_rirufold_cmc/analysis \
  --seeds 42 3407 2026
```

每数据集只有 3 个 seed 时，bootstrap 区间只作描述性稳健性证据，不写成严格的
“统计显著”。

### 6.3 独立测试

```bash
python RiRUFold_ADMM/scripts/evaluate_rirufold_rcp.py \
  --net-name rirufold_cmc \
  --checkpoint result_rirufold_cmc/seed_42/nudt/rirufold_cmc/latest.pkl \
  --dataset nudt --data-root ./RiRUFold_ADMM/datasets --seed 42 \
  --base-size 256 --batch-size 4 --stage-num 5 --hidden-channels 24 \
  --threshold 0.5 --gpu 0 \
  --output result_rirufold_cmc/metrics/seed_42/nudt/rirufold_cmc.json
```

输出包含 mIoU、precision、recall、F1、目标级 Pd、每百万背景像素 Fa、参数量、耗时、
checkpoint SHA-256 和 test split SHA-256。

### 6.4 阶段诊断

```bash
python RiRUFold_ADMM/scripts/diagnose_rirufold_cmc.py \
  --net-name rirufold_cmc \
  --checkpoint result_rirufold_cmc/seed_42/nudt/rirufold_cmc/latest.pkl \
  --image RiRUFold_ADMM/datasets/NUDT-SIRST/test/images/000001.png \
  --stage-num 5 --hidden-channels 24 --base-size 256 --gpu 0 \
  --out-dir result_rirufold_cmc/diagnosis/seed_42/000001
```

应保存 `candidate_score/C/V/side_prior/completion_inner/background/S/W/primal/mu` 图和
`stage_summary.csv`。论文的机制论证图必须来自训练 checkpoint，不可用本地未训练图
替代正式定量结果。

## 7. 正式论文报告要求

- 固定三个数据集 split、epoch、batch、优化器、学习率、stage 数和数据增强。
- 每个配置至少 3 个 seed，主表报告 mean±std；最好增加配对 bootstrap 95% CI。
- 除 mIoU/F1/Pd/Fa 外，报告目标区背景吸收、candidate target/background 均值、运行
  时间和显存。
- 结果图必须同时保留 RCP、CMC、CMC 四个消融和旧 ADMM 基线，不能只选有利样例。
- 若 CMC 只改善分解残差而不改善检测指标，只能写“机制改善”，不能写“检测更优”。

更完整的符号合同、稳定常数、失败条件和作者自检见 `题目分析报告.md` 第 12 节；图表
用途与数据范围见 `figure_contracts.md` 的 q2 部分。
