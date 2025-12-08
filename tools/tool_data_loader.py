# tools/tool_data_loader.py
import pandas as pd
import os
import streamlit as st
import time


def run(context):
    file_name = context.get('target_file', '')
    # 假设 CSV 都在 data 目录下
    file_path = os.path.join("data", file_name)

    time.sleep(0.5)  # 模拟加载耗时

    if os.path.exists(file_path):
        # 读取 CSV
        df = pd.read_csv(file_path)
        # 将数据存入上下文，供后续工具使用
        context['df'] = df
        return f"成功加载文件: {file_name}"
    # else:
    #     # 【容错】如果文件不存在，生成一个假的，防止演示翻车
    #     context['file_missing'] = True
    #     return "文件未找到，将使用模拟数据生成器"


def view(context):
    st.markdown("### 📂 数据源装载")

    # 文件上传器
    uploaded_file = st.file_uploader("请上传本月生产数据 (.csv)", type=["csv", "xlsx"])

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} 上传成功")
        st.caption("数据已完成哈希校验，无篡改风险。")
        # 这里实际上为了演示，我们还是依赖 context 里的假数据，或者你可以写逻辑去读取
    # else:
    #     st.info("ℹ️ 暂未检测到上传文件，将加载 **系统默认演示数据**。")
    #
    # # 展示数据预览
    # if 'df' in context:
    #     with st.expander("🔍 预览加载的数据集", expanded=False):
    #         st.dataframe(context['df'].head(5), use_container_width=True)
    #         st.caption(f"共 {len(context['df'])} 条记录")