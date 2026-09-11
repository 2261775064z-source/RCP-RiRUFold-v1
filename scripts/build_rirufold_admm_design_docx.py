"""Build a Chinese evidence-based RiRUFold ADMM redesign brief."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RiRUFold_ADMM_论据化改进方案.docx"

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "1F2937"
MUTED = "5B6573"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
PALE_YELLOW = "FFF7E0"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int]) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_layout = tbl_pr.first_child_found_in("w:tblLayout")
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tbl_pr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")

    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = tbl.tblGrid
    for grid_col, width in zip(grid.gridCol_lst, widths_dxa):
        grid_col.set(qn("w:w"), str(width))

    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_font(run, size=10.5, bold=False, color=INK, name="Microsoft YaHei") -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_para(p, before=0, after=6, line=1.2, align=None) -> None:
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    if align is not None:
        p.alignment = align


def add_text(doc, text, bold=False, color=INK, size=10.5, before=0, after=6, line=1.2):
    p = doc.add_paragraph()
    set_para(p, before, after, line)
    r = p.add_run(text)
    set_font(r, size=size, bold=bold, color=color)
    return p


def add_bullet(doc, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    set_para(p, after=4, line=1.15)
    r = p.add_run(text)
    set_font(r, size=10.3)
    return p


def add_number(doc, text: str) -> None:
    p = doc.add_paragraph(style="List Number")
    set_para(p, after=4, line=1.15)
    r = p.add_run(text)
    set_font(r, size=10.3)
    return p


def add_heading(doc, text: str, level=1) -> None:
    p = doc.add_paragraph()
    p.style = f"Heading {level}"
    if level == 1:
        set_para(p, before=15, after=7, line=1.0)
        size, color = 15, BLUE
    elif level == 2:
        set_para(p, before=10, after=5, line=1.0)
        size, color = 12.5, DARK_BLUE
    else:
        set_para(p, before=8, after=4, line=1.0)
        size, color = 11, DARK_BLUE
    r = p.add_run(text)
    set_font(r, size=size, bold=True, color=color)
    return p


def add_callout(doc, title: str, body: str, fill=LIGHT_BLUE) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    set_table_geometry(table, [9120])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    set_para(p, before=2, after=2, line=1.15)
    r = p.add_run(f"{title}：")
    set_font(r, size=10.5, bold=True, color=DARK_BLUE)
    r = p.add_run(body)
    set_font(r, size=10.3, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_table(doc, headers: list[str], rows: list[list[str]], widths: list[int], font_size=8.8):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.style = "Table Grid"
    set_table_geometry(table, widths)
    header = table.rows[0]
    set_repeat_table_header(header)
    for cell, text in zip(header.cells, headers):
        set_cell_shading(cell, LIGHT_BLUE)
        p = cell.paragraphs[0]
        set_para(p, after=0, line=1.05, align=WD_ALIGN_PARAGRAPH.CENTER)
        r = p.add_run(text)
        set_font(r, size=font_size, bold=True, color=DARK_BLUE)
    for row_values in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row_values):
            p = cell.paragraphs[0]
            set_para(p, after=0, line=1.05)
            r = p.add_run(text)
            set_font(r, size=font_size, color=INK)
    for i, row in enumerate(table.rows[1:], start=1):
        if i % 2 == 0:
            for cell in row.cells:
                set_cell_shading(cell, "FAFBFC")
    return table


def configure_document(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.62)
    sec.bottom_margin = Inches(0.60)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)
    sec.header_distance = Inches(0.3)
    sec.footer_distance = Inches(0.32)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.0)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.15

    for name in ("Heading 1", "Heading 2", "Heading 3"):
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_para(header, after=0)
    r = header.add_run("RiRUFold ADMM 展开网络改进方案")
    set_font(r, size=8.5, color=MUTED)

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(footer, after=0)
    r = footer.add_run("内部研究设计文档 | 待训练验证后方可写入论文结论")
    set_font(r, size=8, color=MUTED)


def build() -> None:
    doc = Document()
    configure_document(doc)

    p = doc.add_paragraph()
    set_para(p, before=4, after=3, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("RiRUFold 中 ADMM 展开模型的论据化改进方案")
    set_font(r, size=20, bold=True, color=INK)

    p = doc.add_paragraph()
    set_para(p, after=12, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p.add_run("基于 SCLT 张量 ADMM 文献的可借鉴逻辑、现有代码核查与后续验证路径")
    set_font(r, size=10.5, color=MUTED)

    add_callout(
        doc,
        "文档定位",
        "本文件是下一版网络与实验的设计说明，不是论文正文。所有“改进效果”均为待验证假设；只有基于真实训练权重的结果才能用于论文和回复审稿人。",
        PALE_YELLOW,
    )

    add_heading(doc, "一、目标与核心判断", 1)
    add_text(
        doc,
        "现阶段最需要解决的不是继续叠加卷积或注意力模块，而是让 RiRUFold 的网络实现、SRMT/ADMM 公式、图示和消融实验形成一致的论据链。审稿人关注的核心有两点：WLSE 是否只是已有权重的简单相乘；LR_G、SR_G、RW_G、DV_G 是否只是没有严格来源的黑箱卷积。",
    )
    add_callout(
        doc,
        "建议的核心创新表述",
        "提出“背景-稀疏协同演化的 WLSE”：局部结构权重由当前背景估计获得，稀疏增强权重由当前目标估计获得，二者在每个展开阶段耦合后显式控制下一次加权稀疏更新。创新重点不是倒数重加权本身，而是两类信息在 ADMM 状态循环中的分工、耦合和反馈。",
    )

    add_heading(doc, "二、附件文章可借鉴的论证逻辑", 1)
    add_text(
        doc,
        "附件 SCLT 文章是一个面向时空数据补全的 ADMM 张量重建方法，并非 deep unfolding 网络。它不应直接作为 RiRUFold 的展开网络基线，但其方法论非常值得学习：先分清不同信息来源，再把先验放入明确的目标函数与更新序列，最后用组件消融、残差诊断和边界实验验证。该文仍处于 proof/初稿状态，若后续拟引用，需核实正式出版信息。",
    )
    add_table(
        doc,
        ["SCLT 的关键观点", "对 RiRUFold 的可借鉴内容", "对应审稿意见"],
        [
            ["区分不同重建目标及其信息来源", "将背景局部结构与稀疏目标证据分开建模；两者均不应被称为完整目标检测器。", "意见 1"],
            ["将空间先验定义为独立算子", "明确 W_LS 的输入、输出、权重方向及其进入稀疏更新的位置。", "意见 1、5"],
            ["先给目标函数，再推导 ADMM 更新", "在介绍网络前先给出 B、S、W、U、mu 的状态变量和更新顺序。", "意见 2、5"],
            ["反复交换而非一次拼接才构成耦合", "让 B^k 更新 W_LS，S^k 更新 W_SE，W_LSE 再影响 S^(k+1)。", "意见 1"],
            ["记录原始/对偶残差并做停止判断", "输出真实阶段残差曲线，以验证 residual feedback 的作用。", "意见 2"],
            ["实验结论是条件性的，不宣称普适最优", "按背景结构复杂度、SCR、目标尺寸分组，明确 WLSE 的适用场景。", "意见 1、3"],
        ],
        [2700, 4800, 1620],
        8.7,
    )

    add_heading(doc, "三、现有 RiRUFold 代码与论文表述的关键缺口", 1)
    add_table(
        doc,
        ["现有实现", "问题", "对论文解释的影响"],
        [
            ["forward 返回未变化的 D；训练计算 MSE(out_D, data)", "该 MSE 理论上恒为零，背景 B 未被直接监督。", "X = B + S 的分解约束没有进入有效训练目标。"],
            ["LowrankModule 是残差 CNN", "没有显式 SVT 主路径。", "不能直接说 LR_G 等同于低秩近端算子。"],
            ["SparseModule 用动态卷积处理 W*T", "W_LSE 没有直接控制加权软阈值。", "加权 l1 稀疏近端的公式与代码不一致。"],
            ["结构分支来自原始输入 F", "结构图可能仍含目标能量，且没有随 B 的恢复而净化。", "WLSE 容易被理解为静态权重组合。"],
            ["mu 仅初始化一次", "若论文给出 mu^(k+1) 更新，代码与公式矛盾。", "DV_G/罚参数更新的解释不严谨。"],
        ],
        [2520, 3180, 3420],
        8.5,
    )

    add_heading(doc, "四、改进方案一：修复分解训练契约（P0）", 1)
    add_text(doc, "论据：SCLT 的每个通道都进入明确目标函数，并使用残差检验重建一致性。若 RiRUFold 不返回背景估计且背景损失恒为零，则任何“低秩-稀疏分解”解释都缺少训练层面的支撑。")
    add_text(doc, "设计：网络返回最终背景 B^K、目标 S^K 及可选阶段轨迹。使用分解一致性损失 L_dc = ||X - B^K - S^K||_1，以及目标区域外背景损失 L_bg = ||(1-M) ⊙ (B^K-X)||_1。")
    add_text(doc, "代码落点：修改 models/RiRFold.py 的 forward 输出；修改 train.py，删除无效的 MSE(out_D, data)，改为 L_seg + beta_dc L_dc + beta_bg L_bg。")
    add_text(doc, "验证：在真实测试集上绘制 r^k = ||X-B^k-S^k||_F / ||X||_F；完整模型应比无反馈模型有更低或更稳定的最终残差。若没有该现象，不能把 DV_G 的作用归结为分解一致性。", color=DARK_BLUE, bold=True)

    add_heading(doc, "五、改进方案二：背景-稀疏协同演化 WLSE（P1，主创新）", 1)
    add_text(doc, "论据：附件文章最有价值的启发是“不同信息通道在迭代中反复交换”才构成耦合。现有 W_LS 和 W_SE 的乘积如果只来源于原图和当前特征，容易被审稿人视为已知权重的并列使用。")
    add_text(doc, "设计：首先得到 B_avg^(k+1) = (B1^(k+1)+B2^(k+1)+B3^(k+1))/3；再令 W_LS^(k+1)=f_struct(B_avg^(k+1))，W_SE^(k+1)=1/sqrt((S^k)^2+epsilon)，最后 W_LSE^(k+1)=normalize(W_LS^(k+1) ⊙ W_SE^(k+1))。W_LSE 只用于下一次加权稀疏更新。")
    add_text(doc, "可写入论文的解释：W_LS 从逐步净化的背景中描述可能引起虚警的局部结构，W_SE 从当前稀疏分量中保留弱目标证据；二者通过阶段状态相互制约，而不是一次性相乘。")
    add_text(doc, "验证：固定训练策略，比较 Point-WSE、原图静态 WLS + WSE、背景 WLS + 固定 WSE、背景 WLS + 动态 WSE。重点报告高结构复杂度和低 SCR 子集的 Fa/Pd，而不只比较总体 mIoU。", color=DARK_BLUE, bold=True)

    add_heading(doc, "六、改进方案三：显式结构张量先验与小规模学习校准（P2）", 1)
    add_text(doc, "论据：审稿人质疑局部张量掩码只是已有自适应权重的简单组合。要反驳这一点，W_LS 必须具有可检查的结构来源，而不能完全是一个自由 CNN 输出。")
    add_text(doc, "设计：从 B_avg 的梯度构造局部结构张量 J = G_sigma * [[B_x^2, B_xB_y], [B_xB_y, B_y^2]]，再以特征值相干性 c=(lambda_1-lambda_2)/(lambda_1+lambda_2+epsilon) 产生结构惩罚。仅保留斜率、偏置、上下界等少量可学习校准参数。")
    add_text(doc, "验证：对比梯度幅值、显式结构张量、自由 CNN 权重；同时给出目标区和背景区的平均权重、权重比、mIoU/F1/Pd/Fa，并扫描 epsilon、平滑尺度和截断范围。")
    add_callout(doc, "风险边界", "若显式结构张量版本未优于或稳定于自由 CNN，则论文不应再宣称 W_LS 严格来源于结构张量特征值，应改称为“学习到的局部结构估计”。", PALE_YELLOW)

    add_heading(doc, "七、改进方案四：近端锚定的学习校正（P3）", 1)
    add_text(doc, "论据：意见 2 的核心不是要求所有模块都必须为闭式解，而是要求解释学习层为何属于展开过程。把卷积直接称为 proximal operator 缺乏证据；保留显式近端主路径，再让网络学习有界校正更合理。")
    add_text(doc, "设计：低秩主路径为 B_bar^(k+1)=SVT_tau(X-S^k+U^k)；稀疏主路径为 S_bar^(k+1)=Soft(X-B^(k+1)+U^k, lambda_k W_LSE^(k+1)/mu_k)。LR_G 和 SR_G 仅学习校正项，并用 0 到 gamma_max 的门控限制校正幅度。")
    add_text(doc, "验证：比较固定 ADMM、纯学习块、近端锚定块；报告指标、参数量、时间、阶段残差、||B-B_bar|| 和 ||S-S_bar||。只有存在显式近端或近端教师监督时，才可称为 proximal-anchored。")

    add_heading(doc, "八、改进方案五：残差一致性双变量反馈（P4）", 1)
    add_text(doc, "论据：附件文章将 dual variable 作为独立状态，并以原始/对偶残差判断迭代过程。现有 AdaptiveStepGate 有残差输入，但没有残差轨迹，难以证明其不是普通门控。")
    add_text(doc, "设计：R^(k+1)=X-B^(k+1)-S^(k+1)，U^(k+1)=U^k+alpha_k ⊙ R^(k+1)，其中 0<alpha_k<=1。论文中称其为“学习的对偶预条件更新”，不要称完全等价的 ADMM dual update。若不实现 mu 更新，应删除所有 mu^(k+1) 公式。")
    add_text(doc, "验证：比较无反馈、固定步长对偶更新、受限学习预条件器、可选的残差平衡 mu 调度；用最终残差和曲线稳定性支持反馈作用。")

    add_heading(doc, "九、改进方案六：多尺度背景分支赋义（P5，可选）", 1)
    add_text(doc, "论据：B1、B2、B3 若没有不同功能，会被看作重复的黑箱分支。单帧 2D 图像也不应把它们硬称为三个 tensor mode。")
    add_text(doc, "设计：赋予三个分支不同 dilation 或感受野，分别估计小、中、大尺度背景；以 U_B^k=Var(B1^k,B2^k,B3^k) 描述背景不确定性，并仅用于校准 W_LS 的可靠性。")
    add_text(doc, "验证：同尺度分支、多尺度分支、多尺度+不确定性校准的比较。该项属于增强解释性，不应抢占 WLSE 主创新的位置。")

    add_heading(doc, "十、实验与论文材料的对应关系", 1)
    add_table(
        doc,
        ["材料", "应展示的内容", "回答的问题"],
        [
            ["SRMT 目标函数与算法", "变量定义、W_LSE 的插入位置、状态更新顺序、scaled-dual 约定。", "意见 1、2、5"],
            ["Fig. 12 网络图", "B->W_LS，S->W_SE，W_LSE->加权稀疏更新，残差->U 的完整箭头。", "意见 1、2、5"],
            ["算子映射表", "显式近端锚点、学习校正、输入输出变量及诊断量。", "意见 2"],
            ["WLSE 消融表", "静态/动态、未耦合/耦合、真实训练权重下的结果。", "意见 1"],
            ["阶段轨迹图", "B、S、W_LS、W_SE、W_LSE 和原始/对偶残差。", "意见 1、2"],
            ["条件分组与敏感性实验", "背景结构、SCR、目标尺寸、epsilon/lambda/阶段数/权重扰动。", "意见 1、3"],
        ],
        [2250, 4950, 1620],
        8.7,
    )

    add_heading(doc, "十一、推荐执行顺序与结论边界", 1)
    add_number(doc, "先修复 P0：使背景输出、分解损失和阶段残差真实存在。")
    add_number(doc, "实施 P1：用背景条件化 WLS 和显式加权稀疏路径重构 WLSE。")
    add_number(doc, "训练四个 WLSE 变体，完成真实消融与高结构/低 SCR 分组分析。")
    add_number(doc, "实施 P4 并导出残差曲线；据此决定是否保留 dual feedback 的强解释。")
    add_number(doc, "若训练资源允许，再实施 P2 和 P3，形成最强的可解释展开版本。")

    add_callout(
        doc,
        "最终结论边界",
        "本轮最可靠的贡献不应写成“提出了全新的重加权原理”或“卷积严格等同近端算子”。更准确的结论是：RiRUFold 在保留低秩、稀疏、掩码和残差状态的前提下，通过背景-稀疏协同演化的结构约束重加权，改善复杂背景下的弱目标提取；其有效范围应由真实的条件化实验决定。",
        PALE_YELLOW,
    )

    add_heading(doc, "附：可直接用于回复审稿人的简要逻辑", 1)
    add_text(doc, "针对意见 1：承认倒数稀疏重加权本身不是新的正则化原则；本文的新增设计在于背景来源的局部结构先验与阶段性稀疏证据在 SRMT 更新中的状态耦合，并通过静态/动态、单独/耦合的真实训练消融验证。")
    add_text(doc, "针对意见 2：不再将通用卷积直接等同于近端算子，而是以显式 SVT/加权软阈值作为锚点，学习模块仅完成受限校正；同时报告状态变量和原始/对偶残差，说明反馈分支的优化职责。")
    add_text(doc, "针对意见 3：除主表 SOTA 外，补充算子级、机制级、条件化和敏感性实验，明确模型在复杂结构、低 SCR 场景下的有效性边界。")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
