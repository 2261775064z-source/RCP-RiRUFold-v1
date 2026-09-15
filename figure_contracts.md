# RCP-RiRUFold 候选图契约

本轮只有一个研究子问题 `q1`：在不混淆物理稀疏强度与分割 logits 的前提下，
三尺度块低秩共识、可靠性对数池化、逐样本残差反馈及 Blob 支线是否形成可实现、
可诊断、可证伪的新展开方法。所有本地图只用于数据与机制核验；未训练模型的输出
不作为检测精度证据。

| 文件 | 核心结论 | 类别/原型 | 主证据与统计 | 输出与风险控制 |
|---|---|---|---|---|
| `raw_q1_nudt_intensity_ecdf` | 本地 NUDT 测试图的强度与对比度有实际跨度 | raw_q1；单面板 | 前 32 张完整像素的 ECDF；不删异常值 | 6.3 in，SVG+300 DPI PNG；不把图像像素误作独立样本做显著性检验 |
| `raw_q1_target_area_distribution` | 本地测试掩膜中的目标连通域确属小面积对象 | raw_q1；主图+辅助摘要 | 全部 664 张测试掩膜；8 邻域连通域面积，原始点+中位数 | 6.3 in；对数横轴显式标注；零目标图不伪造面积 |
| `raw_q1_synthetic_scene` | 合成门禁同时包含亮点、强边缘和纹理干扰 | raw_q1；同尺度网格 | 合成输入、目标掩膜、排除目标后的强边缘区 | 6.3 in；输入带强度 colorbar，二值图不使用连续色条 |
| `process_q1_residual_trajectory` | 各阶段 primal/dual 残差与逐样本 mu 可追踪 | process_q1；单面板 | 一个固定合成场景的阶段 trace；范数按首阶段归一化 | 6.3 in；迭代折线仅连接真实阶段顺序 |
| `process_q1_patch_consensus` | 三个 patch 分支权重与不一致性均可诊断 | process_q1；主图+辅助证据 | 各阶段三分支空间均值；不一致性像素分布 | 6.3 in；尺度颜色跨图固定，辅助直方图从零开始 |
| `process_q1_threshold_profile` | 可靠性池化和 Blob 支线确实改变局部阈值 | process_q1；单面板 | 穿过目标与强边缘的同一行；三配置最终阈值 | 6.3 in；共享 y 单位，不用双轴 |
| `result_q1_synthetic_decomposition` | 主线输出可分解为背景、物理稀疏强度和独立概率 | result_q1；图像+定量结果 | 固定合成场景；输入/背景共享尺度，稀疏/概率各自标度 | 7.2 in；colorbar 标注单位；注明未训练 |
| `result_q1_mechanism_metrics` | 支线是否越过预注册机制门槛由逐场景数据决定 | result_q1；同尺度定量网格 | 多个合成场景的原始点、中位数及门槛线 | 6.3 in；不以均值柱隐藏分布；不外推检测精度 |
| `result_q1_blob_tradeoff` | 能产生稳定救援的 Blob 强度是否同时违反空场约束 | result_q1；单面板关系图 | 五个 `gamma_g` 的 target 中位响应与空场超阈比例增量 | 6.3 in；显示双门槛，不把不可行点称改进 |
| `result_q1_real_nudt_trace` | 真实 NUDT 图像可跑通完整状态链 | result_q1；图像+定量结果 | 一张固定测试图经 64×64 resize 的 input/B/S/W | 7.2 in；只证明接口与数值链路，不证明性能 |

源数据追溯：真实数据来自 `datasets/NUDT-SIRST/test/{images,masks}`；合成场景由
`scripts/run_rcp_research.py` 在固定种子下生成。数据剖析表记录每张图的尺寸、强度
范围、均值、标准差、目标像素数和连通域数；未执行删行或异常值剔除。

## CMC-RiRUFold 第二研究问题 q2

`q2` 检查“候选观测排除—行随机侧先验—掩蔽低秩补全—正精度共识”是否形成可实现、
可诊断、可否证的第二套展开方法。合成目标面积按本地 NUDT-SIRST 测试掩膜的中位
像素占比约 `39/65536` 校准；所有 result 图仍是固定初始化的未训练机制结果。

| 文件 | 核心结论 | 类别/原型 | 主证据与统计 | 输出与风险控制 |
|---|---|---|---|---|
| `raw_q2_synthetic_challenges` | 合成门禁同时包含数据量级小目标、阶跃和斜纹理 | raw_q2；同尺度网格 | 输入、无目标背景、目标支持、杂波支持 | 7.2 in，SVG+300 DPI PNG；目标面积依据真实数据校准 |
| `raw_q2_candidate_score_profiles` | 候选分数在亮点附近降低直接背景观测权 | raw_q2；单面板 | 固定场景目标行上的 `A/C/V` | 6.3 in；不把 `C` 称分割概率；线型支持灰度辨识 |
| `raw_q2_nudt_local_contrast` | 真实 NUDT 目标与背景的正局部对比分布不同 | raw_q2；ECDF | 前 32 张测试图；目标全像素、背景固定步长抽样 | 6.3 in；不做像素独立显著性检验，不据此选 test 参数 |
| `process_q2_mask_completion_sequence` | 初始侧先验和两次低秩更新均可直接追踪 | process_q2；顺序图 | `C/V/B_A/L^1/L^2` 同一阶段状态 | 7.2 in；只说明数值交换发生，不声称目标单调下降 |
| `process_q2_precision_components` | 三路精度与候选面积日程均显式可导出 | process_q2；双面板 | 每阶段 `p_X/p_L/p_A` 与 mask mean/schedule | 6.3 in；同单位分面，不用双轴 |
| `process_q2_stage_residuals` | residual、mask 和反事实损失随阶段可诊断 | process_q2；双面板 | primal/dual 按首阶段归一化；两个辅助损失原值 | 6.3 in；不从未训练轨迹声称收敛 |
| `result_q2_counterfactual_decomposition` | 背景、物理稀疏量与概率输出保持不同语义 | result_q2；图像网格 | 固定合成场景与无目标背景真值 | 7.2 in；标题显式注明 untrained；不作为检测精度 |
| `result_q2_mechanism_metrics` | mask/side 缺失时目标吸收明显，其他消融保留权衡 | result_q2；原始点+中位数 | 8 个固定合成场景，无删点 | 6.3 in；不用均值柱，不隐藏 `onepass` 接近主线的事实 |
| `result_q2_ablation_tradeoff` | 目标吸收与目标—边缘分离必须联合判断 | result_q2；关系图 | 五个配置的逐场景中位数 | 6.3 in；直接标注，方向写入坐标名；不外推 mIoU/Pd/Fa |

服务器训练完成后由 `scripts/analyze_cmc_server_results.py` 生成下列正式结果图；当前
仓库不预填任何训练精度，也不允许用合成占位数字生成这些文件。

| 文件 | 核心结论 | 类别/原型 | 主证据与统计 | 输出与风险控制 |
|---|---|---|---|---|
| `result_q2_server_main` | CMC 与旧 ADMM、第一方案 RCP 的跨数据集主比较 | result_q2；四指标分面点线图 | 三数据集、三个配对 seed 的 mean±std | 7.2 in；全部完整 test；同 split hash/threshold；不挑 seed |
| `result_q2_server_ablation` | mask、side、inner exchange、stage re-estimation 的训练后作用 | result_q2；四指标分面点线图 | CMC 与四个预注册消融，mean±std | 7.2 in；完整展示五配置；不隐藏 onepass 负结果 |
| `result_q2_server_paired_gain` | CMC 相对 RCP 的逐数据集配对增益方向和不确定性 | result_q2；区间点图 | 三个 seed 的配对差及描述性 bootstrap 95% CI | 7.2 in；正值统一表示 CMC 更好；n=3 不称统计显著 |

q2 原始数据由 `scripts/run_cmc_research.py --seed 20260912` 生成，归档于
`results/cmc_mechanism_metrics.csv`、`cmc_stage_trace.csv`、`cmc_mask_scan.csv`、
`cmc_model_complexity.csv`、`cmc_nudt_profile.csv` 与 `cmc_run_summary.json`。
