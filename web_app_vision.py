import streamlit as st
import pandas as pd
import glob
import os
import jieba.analyse
from sentence_transformers import SentenceTransformer, util
import warnings
from fpdf import FPDF

warnings.filterwarnings('ignore')

# ==========================================
# 1. 网页全局设置
# ==========================================
st.set_page_config(page_title="查重工作台", page_icon="💡", layout="wide")
st.title("智慧城市专利：查重工作台")

# ==========================================
# 2. 加载深度学习语义模型
# ==========================================
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ==========================================
# 3. 核心分析逻辑
# ==========================================
@st.cache_data
def analyze_patents_and_draft(draft_text, target_folder):
    search_path = os.path.join(target_folder, "*.xlsx")
    file_paths = glob.glob(search_path)
    all_data = []

    for file in file_paths:
        filename = os.path.basename(file)
        if filename.startswith("~$") or "报告" in filename:
            continue
        df_temp = pd.read_excel(file).fillna("")
        all_data.append(df_temp)

    if not all_data: return None, None
    df_combined = pd.concat(all_data, ignore_index=True)

    cols = df_combined.columns
    id_col = [c for c in cols if '公开' in c or '公告' in c][0]
    title_col = [c for c in cols if '名称' in c][0]
    abstract_col = [c for c in cols if '摘要' in c][0]
    applicant_col = [c for c in cols if '申请' in c and '人' in c][0]

    df_combined = df_combined.drop_duplicates(subset=[id_col])

    all_text = " ".join(df_combined[title_col].astype(str) + " " + df_combined[abstract_col].astype(str))
    global_keywords = jieba.analyse.extract_tags(all_text, topK=20, withWeight=False, allowPOS=('n', 'vn', 'v'))

    def extract_patent_focus(text):
        return "、".join(jieba.analyse.extract_tags(str(text), topK=4, allowPOS=('n', 'vn', 'v')))

    df_combined['本篇技术侧重点'] = df_combined[abstract_col].apply(extract_patent_focus)

    corpus = (df_combined[title_col].astype(str) + " " + df_combined[abstract_col].astype(str)).tolist()

    model = load_model()
    draft_emb = model.encode(draft_text, convert_to_tensor=True)
    corpus_emb = model.encode(corpus, convert_to_tensor=True)

    cosine_scores = util.cos_sim(draft_emb, corpus_emb)[0].cpu().numpy()

    df_combined['与设想相似度(%)'] = (cosine_scores * 100).round(2)
    df_combined['风险评级'] = df_combined['与设想相似度(%)'].apply(
        lambda s: "🔴 高风险" if s >= 70 else ("🟠 中风险" if s >= 50 else "🟢 低风险")
    )

    result_cols = [id_col, title_col, applicant_col, '本篇技术侧重点', abstract_col, '与设想相似度(%)', '风险评级']
    return global_keywords, df_combined.sort_values(by='与设想相似度(%)', ascending=False)[result_cols]

# ==========================================
# 4. 视觉优化版：PDF 报告生成函数
# ==========================================
def generate_pdf_report(project_name, global_keys, df_result, high_risk, mid_risk):
    pdf = FPDF()
    pdf.add_page()

    try:
        pdf.add_font('Hei', '', 'simhei.ttf', uni=True)
        pdf.set_font('Hei', '', 12)
    except Exception as e:
        st.error("无法加载字体，请确保 `simhei.ttf` 放在代码同级目录下。")
        return None

    # 🟢 支持文字颜色设置的安全打印函数
    def safe_print(text, max_len=45, line_height=7, r=0, g=0, b=0):
        pdf.set_text_color(r, g, b)
        lines = str(text).split('\n')
        for line in lines:
            line = line.strip()
            while len(line) > max_len:
                split_idx = max_len
                # 避免标点符号出现在行首
                if split_idx < len(line) and line[split_idx] in ['，', '。', '、', '；', '：', ',', '.', '”', '）', '】']:
                    split_idx += 1

                pdf.cell(0, line_height, line[:split_idx], ln=True)
                line = line[split_idx:]
            if line:
                pdf.cell(0, line_height, line, ln=True)

    # 1. 报告主标题 (深蓝色)
    pdf.set_text_color(0, 51, 102)
    pdf.set_font('Hei', '', 20)
    pdf.cell(0, 12, f"专利查重分析报告: {project_name.split('.')[1].strip()}", ln=True, align='C')
    pdf.ln(2)

    # 2. 统计汇总区域 (带浅蓝色背景块)
    pdf.set_font('Hei', '', 11)
    global_keys_str = '、'.join(global_keys)
    pdf.set_fill_color(240, 248, 255)  # 浅蓝色 AliceBlue 背景
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(0, 8, f"【同行专利底层逻辑焦点】\n {global_keys_str}", fill=True, border=0)
    pdf.ln(2)

    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, f"比对总数: {len(df_result)} 篇   |   高风险: {high_risk} 篇   |   中风险: {mid_risk} 篇", ln=True)
    pdf.ln(2)

    # 3. 科技感深蓝色主分割线
    pdf.set_draw_color(0, 51, 102)
    pdf.set_line_width(0.6)
    pdf.cell(0, 0, "", border='T', ln=True)
    pdf.set_line_width(0.2) # 恢复线条默认宽度
    pdf.ln(6)

    # 4. 提取风险专利，按不同颜色进行层级排版
    risky_df = df_result[df_result['风险评级'].isin(['🔴 高风险', '🟠 中风险'])]
    abstract_col_name = [c for c in df_result.columns if '摘要' in c][0]
    title_name = [c for c in df_result.columns if '名称' in c][0]
    applicant_name = [c for c in df_result.columns if '申请' in c and '人' in c][0]

    if risky_df.empty:
        pdf.set_font('Hei', '', 14)
        safe_print("✅ 结论：当前草稿具备极高新颖性，未发现高/中度撞车风险！", r=34, g=139, b=34) # 森林绿
    else:
        pdf.set_font('Hei', '', 14)
        safe_print("【重点预警专利清单】", line_height=10, r=0, g=51, b=102)
        pdf.ln(2)

        pdf.set_font('Hei', '', 11)

        for index, row in risky_df.iterrows():
            sim_score = float(row['与设想相似度(%)'])

            # 去除原数据中的 Emoji 图标，避免 PDF 乱码，纯用颜色区分
            is_high_risk = '高' in row['风险评级']
            risk_label = "【高风险】" if is_high_risk else "【中风险】"
            r_val, g_val, b_val = (220, 20, 60) if is_high_risk else (255, 140, 0) # 红 or 橙

            # 字段排版与上色
            safe_print(f"{risk_label} 核心逻辑相似度：{sim_score:.2f}%", r=r_val, g=g_val, b=b_val)
            safe_print(f"名  称：{row[title_name]}", r=0, g=0, b=0)                 # 纯黑
            safe_print(f"申 请 人：{row[applicant_name]}", r=120, g=120, b=120)       # 灰色
            safe_print(f"技术侧重点：{row['本篇技术侧重点']}", r=46, g=139, b=87)        # 墨绿
            safe_print(f"摘  要：{row[abstract_col_name]}", r=40, g=40, b=40)       # 深灰

            pdf.ln(3)
            # 专利之间用极浅的灰色细线分隔
            pdf.set_draw_color(225, 225, 225)
            pdf.cell(0, 0, "", border='T', ln=True)
            pdf.ln(4)

    return bytes(pdf.output())

# ==========================================
# 5. 网页交互界面
# ==========================================
st.sidebar.header("📂 专利项目切换")

project_map = {
    "1. 暴雨内涝水尺检测": "专利分析_水尺识别",
    "2. 寻物时间快速回溯": "专利分析_寻物溯源",
    "3. 摄像头开门状态与角度检测": "专利分析_开门角度",
    "4. 街道场景划分与目标识别优化": "专利分析_盲道识别"
}

project = st.sidebar.selectbox("请选择当前要分析的专利：", list(project_map.keys()))
current_folder = project_map[project]

if project == "1. 暴雨内涝水尺检测":
    default_concept = "一种面向复杂视角与遮挡环境的城市内涝水尺识别方法。"
elif project == "2. 寻物时间快速回溯":
    default_concept = "一种基于时序差分矩阵与历史动态采样的监控遗失物快速回溯方法。"
elif project == "3. 摄像头开门状态与角度检测":
    default_concept = "一种多角度监控的房门开启状态与转动角度动态检测方法。"
else:
    default_concept = "一种基于图像分割，再根据检测出来的杂物坐标算出来他是占道经营、店外经营、还是流动商贩方法。"

st.sidebar.header("📝 录入与修改设想")
st.sidebar.info(f"系统已切换至【{project}】")
my_draft = st.sidebar.text_area("您的核心技术设想：", value=default_concept, height=280)

if st.sidebar.button(f"🚀 查重: {project.split('.')[1]}"):
    if not os.path.exists(current_folder):
        st.error(f"找不到数据文件夹：`{current_folder}`。请确保您已在代码同级目录下创建了该文件夹，并将 Excel 放入其中。")
    else:
        with st.spinner(f"正在读取 {current_folder} 中的专利数据，并进行比对..."):
            global_keys, df_result = analyze_patents_and_draft(my_draft, current_folder)

        if df_result is not None:
            st.markdown(f"### 📊 【{project.split('.')[1].strip()}】重点方向关键词")
            st.info(f"**同行专利的底层逻辑焦点：** { ' 、 '.join(global_keys) }")

            st.markdown("---")
            st.markdown(f"### 📑 比堪清单 ({current_folder})")

            high_risk = len(df_result[df_result['风险评级'] == '🔴 高风险'])
            mid_risk = len(df_result[df_result['风险评级'] == '🟠 中风险'])

            col1, col2, col3 = st.columns(3)
            col1.metric("本地数据库对比总数", f"{len(df_result)} 篇")
            col2.metric("🔴 核心逻辑相似预警", f"{high_risk} 篇")
            col3.metric("🟠 局部方案重合风险", f"{mid_risk} 篇")

            abstract_col_name = [c for c in df_result.columns if '摘要' in c][0]
            st.dataframe(df_result.drop(columns=[abstract_col_name]), use_container_width=True, height=350)

            # ==========================================
            # 生成并展示 PDF 下载按钮
            # ==========================================
            st.markdown("---")
            col_btn, _ = st.columns([1, 2])
            with col_btn:
                pdf_bytes = generate_pdf_report(project, global_keys, df_result, high_risk, mid_risk)
                if pdf_bytes:
                    st.download_button(
                        label="📄 一键导出分析报告 (PDF)",
                        data=pdf_bytes,
                        file_name=f"{project.split('.')[1].strip()}_专利查重报告.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning("⚠️ 若需下载PDF报告，请将 `simhei.ttf` 字体文件放置在代码同一目录下。")
            st.markdown("---")

            risky_df = df_result[df_result['风险评级'].isin(['🔴 高风险', '🟠 中风险'])]

            if not risky_df.empty:
                st.markdown(f"### 📖 重点预警：全部 {len(risky_df)} 篇高/中风险专利摘要")
                display_df = risky_df
            else:
                st.markdown("### 📖 相似度最高前 10 篇专利摘要速览")
                st.success("✅ 当前草稿具备极高新颖性，未发现高/中度撞车风险！")
                display_df = df_result.head(10)

            for index, row in display_df.iterrows():
                title_name = [c for c in display_df.columns if '名称' in c][0]
                with st.expander(f"{row['风险评级']} (相似度 {row['与设想相似度(%)']}%) - {row[title_name]}"):
                    st.markdown(f"**申请人：** {row[[c for c in display_df.columns if '申请' in c and '人' in c][0]]}")
                    st.markdown(f"**核心侧重点：** `{row['本篇技术侧重点']}`")
                    st.markdown("**摘要内容：**")
                    st.write(row[abstract_col_name])

        else:
            st.warning(f"文件夹 `{current_folder}` 中目前是空的，请将您在专利网下载的 Excel 表格放入其中。")
else:
    st.info("👈 请在左侧选择研究课题，并在建立好对应文件夹后，点击启动按钮。")