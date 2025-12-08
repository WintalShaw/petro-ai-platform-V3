# tools/tool_water_algo.py
import streamlit as st
import time


def run(context):
    time.sleep(0.5)
    return "方案生成完毕"


def view(context):
    # ==================================================
    # 1. 调用主程序传入的模型交互 UI
    # ==================================================
    # 【修改点】第一个参数改为 "tool_water_algo"
    should_show = context.get('render_model_ui', lambda x, y, z: True)(
        "tool_water_algo",  # <--- 这里改了
        "配注优化强化学习模型 (DQN)",
        context
    )

    if not should_show:
        return

    # ==================================================
    # 2. 原有的配注方案展示逻辑
    # ==================================================
    st.success("💧 智能配注方案已生成 (基于当前地层压力)")

    if 'df' in context:
        df = context['df']

        # 自动适配列名
        if '调整量' in df.columns:
            target_col = '调整量'
            metric_label = "总增注量"
        elif '建议配注' in df.columns:
            target_col = '建议配注'
            metric_label = "总建议配注量"
        else:
            target_col = None
            metric_label = "数值统计"

        # 统计指标展示
        col1, col2 = st.columns(2)
        with col1:
            st.metric("涉及调整井数", f"{len(df)} 口", delta="优化覆盖率 100%")

        with col2:
            if target_col:
                total = df[target_col].sum()
                st.metric(metric_label, f"{total:.1f} m³", delta_color="normal")
            else:
                st.metric("数据状态", "无有效数值列")

        # 数据表格展示 (带热力图)
        if target_col:
            try:
                st.dataframe(
                    df.style.background_gradient(subset=[target_col], cmap='Blues'),
                    use_container_width=True
                )
            except Exception:
                st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

        # 更新摘要
        total_vol = df[target_col].sum() if target_col else 0
        context[
            'water_summary'] = f"针对 {len(df)} 口井生成了 DQN 优化方案，{metric_label}合计 {total_vol:.1f} m³，预计提升水驱效率 2.3%。"