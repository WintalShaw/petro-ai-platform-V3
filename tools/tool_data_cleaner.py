# tools/tool_data_cleaner.py
import streamlit as st
import time
import random


def run(context):
    time.sleep(1.5)  # 假装耗时
    return "清洗完成"


def view(context):
    # --- 1. 生成动态随机数据 ---
    # 模拟填充了几个空值 (0~5个)
    null_count = random.randint(0, 5)

    # 模拟剔除了多少个离群点 (5~35个，模拟去噪)
    outlier_count = random.randint(5, 35)

    # 模拟数据质量评分 (96.0 ~ 99.9)
    # 用 uniform 生成浮点数
    quality_score = random.uniform(96.0, 99.9)

    # --- 2. 界面渲染 ---
    st.success("✅ 数据清洗引擎执行完毕")

    # 使用 f-string 将随机数填入字符串
    st.caption(f"已智能填充空值: {null_count} | 已剔除离群噪点: {outlier_count} | 🛡️ 数据质量评分: {quality_score:.1f}")

    # (可选) 加一个更直观的进度条展示质量
    # st.progress(int(quality_score), text="质量健康度")
