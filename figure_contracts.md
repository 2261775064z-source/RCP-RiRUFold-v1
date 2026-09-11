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
