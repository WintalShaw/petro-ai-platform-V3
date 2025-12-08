# tools/tool_trend_algo.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from matplotlib import font_manager


# --- 字体辅助函数 (保持不变) ---
def get_chinese_font():
    """
    加载项目根目录下的 msyh.ttc 字体
    """
    font_path = "msyh.ttc"
    if not os.path.exists(font_path):
        return font_manager.FontProperties(family='Microsoft YaHei')
    return font_manager.FontProperties(fname=font_path)


def run(context):
    # 后端模拟运行逻辑
    time.sleep(0.5)
    return "预测完成"


def view(context):
    # ==================================================
    # 1. 调用主程序传入的模型交互 UI
    # ==================================================
    # 【修改点】第一个参数改为 "tool_trend_algo" (工具ID)，而不是 "model_trend"
    should_show = context.get('render_model_ui', lambda x, y, z: True)(
        "tool_trend_algo",  # <--- 这里改了
        "产量趋势预测模型 (LSTM-V2)",
        context
    )

    # 如果用户还在决策或训练中，中断渲染
    if not should_show:
        return

    # ==================================================
    # 2. 原有的预测图表渲染逻辑
    # ==================================================
    st.info("📉 正在渲染未来产量趋势预测曲线...")

    zh_font = get_chinese_font()

    if 'df' in context:
        df = context['df']

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # --- 绘图逻辑 ---
        fig, ax = plt.subplots(figsize=(10, 4))

        # 绘制预测线
        ax.plot(df['date'], df['predicted_yield'],
                label='AI 预测趋势',
                color='#d62728',
                linewidth=2.5,
                marker='o',
                markersize=4,
                linestyle='-')

        # 设置中文
        ax.set_title(f"{context.get('month')}月 全周期产量推演 (AI Predicted)",
                     fontsize=12, fontproperties=zh_font)
        ax.legend(loc='upper right', prop=zh_font)
        ax.set_xlabel("预测时间轴", fontproperties=zh_font)
        ax.set_ylabel("日产量 (吨)", fontproperties=zh_font)

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.fill_between(df['date'], df['predicted_yield'], alpha=0.1, color='red')

        st.pyplot(fig)

        # 生成摘要
        if not df.empty:
            min_val = df['predicted_yield'].min()
            max_val = df['predicted_yield'].max()
            context['trend_summary'] = f"预计全月产量将在 {min_val}~{max_val} 吨区间运行，呈现平稳缓降趋势。"
        else:
            context['trend_summary'] = "数据不足，无法生成摘要。"