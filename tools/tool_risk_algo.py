# tools/tool_risk_algo.py
import streamlit as st
import time


def run(context):
    time.sleep(0.5)
    return "风险扫描完成"


def view(context):
    # ==================================================
    # 1. 调用主程序传入的模型交互 UI
    # ==================================================
    # 【修改点】第一个参数改为 "tool_risk_algo"
    should_show = context.get('render_model_ui', lambda x, y, z: True)(
        "tool_risk_algo",   # <--- 这里改了
        "风险预警分类器 (XGBoost)",
        context
    )

    if not should_show:
        return

    # ==================================================
    # 2. 原有的风险展示逻辑
    # ==================================================
    if 'df' in context:
        df = context['df']

        # 简单判断是否有风险
        has_risk = False
        if '风险值' in df.columns and df['风险值'].max() > 0.8:
            has_risk = True

        if has_risk:
            st.warning("⚠️ 发现潜在生产风险点！(置信度 > 85%)")
        else:
            st.success("✅ 当前生产状况健康，未发现显著异常。")

        # 1. 风险统计图
        if '风险类型' in df.columns:
            st.caption("风险类型分布统计")
            risk_counts = df['风险类型'].value_counts()
            st.bar_chart(risk_counts, color="#ff4b4b")

        # 2. 高风险列表
        st.write("🔴 **重点关注井号清单**")
        if '风险值' in df.columns:
            # 筛选高风险
            high_risk = df[df['风险值'] > 0.6].copy()
            # 格式化一下风险值显示
            high_risk['风险值'] = high_risk['风险值'].apply(lambda x: f"{x * 100:.1f}%")

            st.dataframe(high_risk, use_container_width=True)
            context['risk_summary'] = f"扫描发现 {len(high_risk)} 口井存在潜在风险，建议优先排查套损问题。"
        else:
            st.dataframe(df)
            context['risk_summary'] = "整体风险可控，无高等级预警。"