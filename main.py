import streamlit as st
import importlib
import time
import random
import json
import os
import datetime
import graphviz
from agent_brain import plan_workflow

st.set_page_config(page_title="油气生产一体化智能系统", layout="wide", page_icon="🛢️")

USER_DB_FILE = "users.json"
REPORT_DB_FILE = "reports.json"

TOOL_META = {
    "tool_data_loader": {"name": "多源数据集成加载", "icon": "📂"},
    "tool_data_cleaner": {"name": "异常值清洗引擎", "icon": "🧹"},
    "tool_feature_eng": {"name": "时序特征工程构建", "icon": "🔧"},
    "tool_correlation": {"name": "多维因子关联分析", "icon": "🕸️"},
    "tool_model_inference": {"name": "深度学习模型推理", "icon": "🧠"},
    "tool_trend_algo": {"name": "产量趋势预测算法", "icon": "📈"},
    "tool_risk_algo": {"name": "生产风险扫描引擎", "icon": "⚠️"},
    "tool_water_algo": {"name": "智能配注优化模型", "icon": "💧"},
    "tool_report_gen": {"name": "AI 决策报告生成", "icon": "📝"},
    "tool_approval_flow": {"name": "自动审批流程推送", "icon": "📤"},
}

MODELS_LIST = [
    {"id": "model_trend", "name": "产量趋势预测模型 (LSTM-V2)", "last_update": "2024-05-20"},
    {"id": "model_risk", "name": "风险预警分类器 (XGBoost)", "last_update": "2024-06-01"},
    {"id": "model_water", "name": "配注优化强化学习模型 (DQN)", "last_update": "2024-04-15"},
]

# --- CSS 样式 ---
st.markdown("""
<style>
    .stSpinner > div {border-top-color: #0f52ba !important;}
    .element-container {margin-bottom: 10px;}
    .notification-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
        background-color: #ffeaea;
    }
    div[data-testid="stExpander"] details {
        border: 1px solid #ff4b4b;
        border-radius: 5px;
        background-color: #fff5f5;
    }
</style>
""", unsafe_allow_html=True)



def init_db():
    default_users = {
        "mr.gong": {"password": "123456", "role": "admin", "model_states": {}},  # 新增 model_states
        "user": {"password": "123", "role": "user", "model_states": {}}  # 新增 model_states
    }

    with open(USER_DB_FILE, "r", encoding='utf-8') as f:
        users = json.load(f)
        changed = False
        for u in users:
            # 补全 model_states
            if "model_states" not in users[u]:
                users[u]["model_states"] = {}
                changed = True
            # [新增] 补全 history
            if "history" not in users[u]:
                users[u]["history"] = []
                changed = True
        if changed:
            save_data(USER_DB_FILE, users)


    if not os.path.exists(USER_DB_FILE):
        # 加上 encoding='utf-8'
        with open(USER_DB_FILE, "w", encoding='utf-8') as f:
            json.dump(default_users, f, ensure_ascii=False, indent=4)
    else:
        # 读取也要加
        with open(USER_DB_FILE, "r", encoding='utf-8') as f:
            users = json.load(f)
        if "mr.gong" not in users:
            users["mr.gong"] = default_users["mr.gong"]
            with open(USER_DB_FILE, "w", encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=4)

    with open(USER_DB_FILE, "r", encoding='utf-8') as f:
        users = json.load(f)
        changed = False
        for u in users:
            if "model_states" not in users[u]:
                users[u]["model_states"] = {}
                changed = True
        if changed:
            save_data(USER_DB_FILE, users)

    if not os.path.exists(REPORT_DB_FILE):
        # 加上 encoding='utf-8'
        with open(REPORT_DB_FILE, "w", encoding='utf-8') as f:
            json.dump([], f)


def load_data(file):
    # 加上 encoding='utf-8'
    with open(file, "r", encoding='utf-8') as f:
        return json.load(f)


def save_data(file, data):
    with open(file, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



HISTORY_BASE_DIR = "user_history"


def ensure_user_history_dir(username):
    """确保用户的历史记录文件夹存在"""
    user_dir = os.path.join(HISTORY_BASE_DIR, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir


def save_session_to_disk(username, session_data, history_id=None, custom_title=None):
    """保存当前会话到磁盘，并更新 users.json 中的索引"""
    user_dir = ensure_user_history_dir(username)

    # 1. 确定 ID 和 文件名
    if not history_id:
        history_id = str(int(time.time()))

    filename = f"{history_id}.json"
    file_path = os.path.join(user_dir, filename)

    # 2. 保存会话内容文件
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)

    # 3. 更新用户的历史索引 (users.json)
    users = load_data(USER_DB_FILE)
    if "history" not in users[username]:
        users[username]["history"] = []

    # 如果没提供标题，尝试从数据中生成
    if not custom_title:
        # 尝试获取任务名，没有则取第一句对话
        task_name = session_data.get("current_context", {}).get("task_name")
        if not task_name and len(session_data.get("messages", [])) > 1:
            task_name = session_data["messages"][1]["content"][:10] + "..."
        custom_title = task_name or "未命名会话"

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    title_display = f"📅 {timestamp_str} | {custom_title}"

    # 检查是否已存在该 ID (如果是更新旧存档)
    existing_idx = next((i for i, item in enumerate(users[username]["history"]) if item["id"] == history_id), -1)

    history_item = {
        "id": history_id,
        "title": title_display,  # 存完整的显示标题
        "file_path": file_path,
        "updated_at": timestamp_str
    }

    if existing_idx != -1:
        # 更新旧记录 (移到最前)
        users[username]["history"].pop(existing_idx)
        users[username]["history"].insert(0, history_item)
    else:
        # 插入新记录
        users[username]["history"].insert(0, history_item)

    save_data(USER_DB_FILE, users)
    return history_id


def load_session_from_disk(username, history_id):
    """从磁盘读取会话内容"""
    users = load_data(USER_DB_FILE)
    history_list = users.get(username, {}).get("history", [])

    # 找到对应的文件路径
    target_item = next((item for item in history_list if item["id"] == history_id), None)
    if target_item and os.path.exists(target_item["file_path"]):
        with open(target_item["file_path"], "r", encoding='utf-8') as f:
            return json.load(f)
    return None


def submit_report_to_manager(username, task_name, context):
    """普通用户提交报告"""
    reports = load_data(REPORT_DB_FILE)
    new_report = {
        "id": f"RPT-{int(time.time())}",
        "submitter": username,
        "task_name": task_name,
        "submit_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
         "status": "pending",
        "feedback": "",
        "file_path": context.get('target_file', '未知文件.csv'),
        # 简单模拟报告内容
        "summary": context.get('trend_summary') or context.get('risk_summary') or context.get(
            'water_summary') or "自动生成的分析报告"
    }
    reports.insert(0, new_report)  # 最新在最前
    save_data(REPORT_DB_FILE, reports)


def init_session():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None  # user 或 admin
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"



def render_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("system-logo.png", width=80)
        st.title("AI 油气生产指挥系统")
        st.caption("Enterprise Edition V3.2")

        tab1, tab2 = st.tabs(["🔐 账号登录", "📝 员工注册"])
        users = load_data(USER_DB_FILE)

        # --- 登录 ---
        with tab1:
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pass")
            if st.button("登录", use_container_width=True):
                if username in users and users[username]['password'] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = users[username].get('role', 'user')

                    # 路由判断
                    if st.session_state.role == 'admin':
                        st.session_state.current_page = "manager_dashboard"
                        st.success(f"欢迎宫老师！正在进入审批工作台...")
                    else:
                        st.session_state.current_page = "analysis"
                        st.success(f"登录成功！正在加载 {username} 的工作环境...")

                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

        # --- 注册 (屏蔽 mr.wang) ---
        with tab2:
            new_user = st.text_input("设置用户名", key="reg_user")
            new_pass = st.text_input("设置密码", type="password", key="reg_pass")
            if st.button("注册并初始化", use_container_width=True):
                if new_user.lower() == "mr.wang" or "admin" in new_user.lower():
                    st.error("❌ 无法注册管理层账号，请联系IT部门。")
                elif new_user in users:
                    st.error("用户已存在")
                elif new_user and new_pass:
                    with st.spinner(f"正在为 {new_user} 分配独立空间..."):
                        time.sleep(1)
                        users[new_user] = {
                            "password": new_pass,
                            "role": "user",
                            "model_path": f"/usr/local/ai_models/{new_user}/"
                        }
                        save_data(USER_DB_FILE, users)
                    st.success("注册成功！请登录。")


@st.dialog("🧠 AI 模型与工具库全景")
def show_model_library_modal():
    st.caption("查看平台现有的公共算法模型及您训练的专属模型。")

    # 1. 搜索框
    search_query = st.text_input("🔍 搜索模型名称...",
                                 placeholder="输入关键字查找，例如 '预测' 或 '风险'").strip().lower()

    # 2. 准备数据
    users = load_data(USER_DB_FILE)
    # 获取当前用户的模型状态字典
    user_states = users.get(st.session_state.username, {}).get("model_states", {})

    public_tools = []
    private_tools = []

    # ================= 修改开始：定义映射关系 =================
    # 核心修复：建立 工具ID (Tool ID) -> 数据库存储的模型ID (DB Key) 的映射
    # 只有这三个是深度学习模型，其他工具 ID 保持不变
    tool_map = {
        "tool_trend_algo": "model_trend",
        "tool_risk_algo": "model_risk",
        "tool_water_algo": "model_water"
    }
    # ========================================================

    # 3. 遍历并分类
    for tool_id, meta in TOOL_META.items():
        # 搜索过滤逻辑
        if search_query and (search_query not in meta['name'].lower()):
            continue

        # ================= 修改开始：使用映射查找状态 =================
        # 如果这个工具在映射表中（说明它是深度模型），就用映射后的 ID 去查库
        # 否则（比如数据清洗工具），直接用工具 ID 查
        db_key = tool_map.get(tool_id, tool_id)

        # 判断状态 (用 db_key 去查)
        state = user_states.get(db_key, "untrained")
        # ==========================================================

        tool_info = {
            "name": meta['name'],
            "icon": meta['icon'],
            "id": tool_id,  # 显示时还是显示工具ID，或者显示db_key也可以
            "desc": "标准算法" if state != "private" else "已针对您的数据微调"
        }

        if state == "private":
            private_tools.append(tool_info)
        else:
            public_tools.append(tool_info)

    # 4. 展示分栏 (Tabs)
    tab1, tab2 = st.tabs([f"🏛️ 公共通用库 ({len(public_tools)})", f"🔐 我的专属库 ({len(private_tools)})"])

    with tab1:
        if not public_tools:
            st.info("没有找到匹配的公共模型。")
        for tool in public_tools:
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(f"<h1>{tool['icon']}</h1>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{tool['name']}**")
                    st.caption(f"ID: {tool['id']}")
                    st.caption("🟢 状态: 公共可用")

    with tab2:
        if not private_tools:
            st.info("您还没有私有化部署的模型。请在对话分析中对模型进行训练和保存。")
        for tool in private_tools:
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.markdown(f"<h1>{tool['icon']}</h1>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**{tool['name']}**")
                    st.caption(f"ID: {tool['id']}")  # 这里也可以显示映射后的ID
                    st.caption("✨ 状态: **已私有化 (Private)**")


def render_sidebar():
    with st.sidebar:
        # --- 1. 消息通知区域 (置顶) ---
        if st.session_state.role == 'user':
            reports = load_data(REPORT_DB_FILE)
            # 筛选：当前用户 + 状态是 rejected
            rejected_list = [r for r in reports if
                             r['submitter'] == st.session_state.username and r.get('status') == 'rejected']

            if rejected_list:
                st.error(f"🔔 您有 {len(rejected_list)} 条驳回通知")
                with st.expander("查看驳回详情", expanded=True):
                    for r in rejected_list:
                        st.markdown(f"""
                                <div class="notification-box">
                                    <small>任务: {r['task_name']}</small><br>
                                    <strong>❌ 意见: {r.get('feedback', '无')}</strong>
                                </div>
                                """, unsafe_allow_html=True)

                    # 清除通知按钮
                    if st.button("我知道了 (清除通知)", key="cls_msg", use_container_width=True):
                        # 逻辑：保留那些【不是(当前用户且被驳回)】的报告
                        new_reports = [
                            x for x in reports
                            if not (x['submitter'] == st.session_state.username and x.get('status') == 'rejected')
                        ]
                        save_data(REPORT_DB_FILE, new_reports)
                        st.rerun()
                st.divider()

        st.title("🛢️ 智能生产平台")

        # --- 2. 用户信息与模型库按钮 ---
        role_icon = "👨‍💼" if st.session_state.role == 'admin' else "👷"
        role_name = "生产经理" if st.session_state.role == 'admin' else "生产工程师"

        with st.container(border=True):
            st.write(f"{role_icon} **用户**: {st.session_state.username}")
            st.caption(f"身份: {role_name}")

            # [新增] 弹窗触发按钮
            if st.button("📚 查看模型库", use_container_width=True):
                show_model_library_modal()

        st.divider()

        # --- 3. 历史存档 (只读) ---
        st.markdown("### 🕒 历史存档")

        users = load_data(USER_DB_FILE)
        user_history = users.get(st.session_state.username, {}).get("history", [])

        # 如果没有历史，显示默认
        if not user_history:
            st.caption("暂无历史记录")

        # 遍历显示历史按钮
        for item in user_history:
            # 样式优化：如果当前正在编辑这个存档，高亮显示
            if st.session_state.get("current_history_id") == item["id"]:
                btn_type = "primary"
                label = f"📂 {item['title']} (编辑中)"
            else:
                btn_type = "secondary"
                label = item['title']

            if st.button(label, key=f"hist_{item['id']}", type=btn_type, use_container_width=True):
                # 因为没有保存文件，这里仅做提示
                st.toast(f"📄 这是一个归档记录：{item['title']}")

        st.divider()

        # --- 4. 导航菜单 ---
        if st.session_state.role == 'user':
            st.markdown("### 🧭 导航菜单")
            if st.button("📊 生产分析", use_container_width=True,
                         type="primary" if st.session_state.current_page == "analysis" else "secondary"):
                st.session_state.current_page = "analysis"
                st.rerun()
            if st.button("🔧 参数微调", use_container_width=True,
                         type="primary" if st.session_state.current_page == "training" else "secondary"):
                st.session_state.current_page = "training"
                st.rerun()

            if st.session_state.current_page == "analysis":
                st.write("")
                # ==========================================
                # [保留] 清空并归档 (无报错版本)
                # ==========================================
                if st.button("🗑️ 清空并归档", use_container_width=True):
                    # 1. 只有当有过对话时才记录
                    if "messages" in st.session_state and len(st.session_state.messages) > 1:

                        # --- A. 获取任务名作为标题 ---
                        ctx = st.session_state.get("current_context", {})
                        if ctx and "task_name" in ctx:
                            task_name = ctx["task_name"]
                        else:
                            try:
                                # 取用户说的第一句话
                                task_name = st.session_state.messages[1]["content"][:10] + "..."
                            except:
                                task_name = "未命名任务"

                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        title_display = f"📅 {timestamp} | {task_name}"

                        # --- B. 更新左侧历史列表 (写入 users.json) ---
                        users = load_data(USER_DB_FILE)
                        username = st.session_state.username

                        if "history" not in users[username]:
                            users[username]["history"] = []

                        # 创建记录项 (注意：这里我们故意不保存 file_path 对应的物理文件)
                        new_record = {
                            "id": str(int(time.time())),  # 唯一ID
                            "title": title_display,  # 显示的标题
                            "file_path": "",  # 留空
                            "updated_at": timestamp
                        }

                        # 插入到最前面
                        users[username]["history"].insert(0, new_record)
                        save_data(USER_DB_FILE, users)

                        st.toast("✅ 历史记录已归档")
                        time.sleep(0.5)

                    # 2. 彻底清空工作台
                    st.session_state.messages = [
                        {"role": "assistant", "content": "您好！我是您的专属AI生产指挥官。请告诉我要分析的任务。"}]
                    st.session_state.current_workflow = None
                    st.session_state.current_context = None
                    st.session_state.workflow_step = 0
                    st.session_state.workflow_finished = False
                    st.session_state.current_history_id = None

                    # 清除临时状态
                    keys_to_del = [k for k in st.session_state.keys() if
                                   k.startswith("submitted_") or k.startswith("trained_")]
                    for k in keys_to_del:
                        del st.session_state[k]

                    # 3. 刷新页面
                    st.rerun()

            # [删除] 原来的“已激活工具”列表已移除

        st.divider()
        # --- 5. 退出登录 ---
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.current_page = "login"
            st.rerun()


def render_manager_page():
    st.title("👨‍💼 生产经理审批工作台")
    st.caption(f"当前用户: 宫老师 (mr.gong) | 部门: 生产运行科 | 权限: Level-5")

    # --- [新增] 统计数据持久化逻辑 ---
    STATS_FILE = "manager_stats.json"

    # 1. 初始化或读取统计数据
    if not os.path.exists(STATS_FILE):
        # 如果文件不存在，给一个初始值（比如本周已经处理了15个），假装系统一直在运行
        stats_data = {"processed_count": 15}
        with open(STATS_FILE, "w", encoding='utf-8') as f:
            json.dump(stats_data, f)
    else:
        with open(STATS_FILE, "r", encoding='utf-8') as f:
            stats_data = json.load(f)

    current_processed = stats_data.get("processed_count", 15)

    # 2. 读取报告数据
    reports = []
    if os.path.exists("reports.json"):
        with open("reports.json", "r", encoding='utf-8') as f:
            reports = json.load(f)

    # 3. 顶部仪表盘
    pending_count = len(reports)
    col1, col2, col3 = st.columns(3)
    col1.metric("待处理审批", pending_count, delta="实时更新" if pending_count > 0 else "无积压", delta_color="inverse")

    # 【修改点】这里不再随机，而是读取真实记录的数字
    col2.metric("本周已处理", current_processed, delta="+1" if 'just_processed' in st.session_state else None)

    # 清除刚才的 +1 动画状态
    if 'just_processed' in st.session_state:
        del st.session_state['just_processed']

    col3.metric("系统健康度", "98.5%")

    st.divider()

    # 4. 如果没有报告
    if not reports:
        st.container(border=True).info("🍵 当前工作台空空如也，您可以喝杯茶休息一下。")
        if st.button("🔄 刷新数据"):
            st.rerun()
        return

    # 5. 待办列表 (遍历显示)
    st.subheader(f"📋 待办事项 ({pending_count})")

    # 使用副本遍历，防止删除时索引错位
    # 注意：这里直接用enumerate可能会有删除索引问题，演示版简单处理即可
    for i, report in enumerate(reports):
        if report.get('status') != 'pending':
            continue
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])

            # 左侧：报告详情
            with c1:
                st.markdown(f"### 📑 {report['task_name']}")
                st.caption(
                    f"提交人: **{report['submitter']}** | 提交时间: {report['submit_time']} | ID: {report['id']}")

                # 摘要框
                st.text_area("AI 分析结论摘要", report['summary'], height=60, disabled=True, key=f"txt_{i}")

                # 模拟附件下载
                # 模拟附件下载
                # 【修复】优先读 file_path，读不到就读 file_name，再读不到就给默认值
                safe_name = report.get('file_path', report.get('file_name', 'report.csv'))

                st.download_button(
                    label="📥 下载完整数据包 (.csv)",
                    data="Simulated Content",
                    file_name=safe_name,
                    key=f"dl_{i}"
                )

            # 右侧：审批操作
            with c2:
                st.write("")  # 占位
                st.write("")

                # --- 定义一个更新统计数据的内部函数 ---
                def update_stats():
                    new_count = current_processed + 1
                    with open(STATS_FILE, "w", encoding='utf-8') as f:
                        json.dump({"processed_count": new_count}, f)
                    st.session_state['just_processed'] = True  # 触发UI上的绿色小箭头

                # --- 同意按钮 ---
                if st.button("✅ 批准执行", key=f"app_{i}", type="primary", use_container_width=True):
                    # 1. 更新统计 (数字+1)
                    update_stats()

                    # 2. 删除报告
                    reports.pop(i)
                    with open("reports.json", "w", encoding='utf-8') as f:
                        json.dump(reports, f, ensure_ascii=False, indent=4)

                    st.toast("审批已通过！报告已归档。")
                    time.sleep(0.5)
                    st.rerun()

                # --- 驳回按钮 ---
                if st.button("❌ 驳回重做", key=f"rej_{i}", use_container_width=True):
                    # 1. 更新统计 (数字+1)
                    update_stats()

                    # 2. 【关键修改】不删除，而是改状态，并写入反馈意见
                    reports[i]['status'] = 'rejected'
                    # 这里为了演示简单写死，你也可以加个 st.text_input 让经理输入
                    reports[i]['feedback'] = "数据特征工程存在异常，请重新检查相关性分析结果。"

                    # 3. 保存回文件
                    save_data(REPORT_DB_FILE, reports)

                    st.toast("已驳回！通知已发送给提交人。")
                    time.sleep(0.5)
                    st.rerun()


def render_training_page():
    st.title("🔧 工具参数更新与微调中心")
    st.caption(f"当前工作空间: {st.session_state.username}@cluster-08")

    col_list, col_detail = st.columns([1, 2])

    # 左侧：选择模型
    with col_list:
        st.subheader("🛠️ 已部署模型库")
        selected_model = st.radio(
            "选择要更新的工具/模型:",
            [m["name"] for m in MODELS_LIST],
            label_visibility="collapsed"
        )

        # 找到对应的模型ID
        model_info = next(item for item in MODELS_LIST if item["name"] == selected_model)

        st.info(f"上次更新时间: {model_info['last_update']}")
        st.warning("提示: 更新参数将触发热加载，不影响当前生产任务。")

    # 右侧：上传与更新面板
    with col_detail:
        with st.container(border=True):
            st.subheader(f"🚀 更新向导: {selected_model}")

            # 步骤 1: 上传
            st.markdown("**Step 1: 上传增量训练数据 (CSV)**")
            uploaded_file = st.file_uploader("拖拽文件到此处", type=["csv"])

            # 步骤 2: 验证与更新
            if uploaded_file is not None:
                st.success(f"✅ 文件已校验: {uploaded_file.name} (12.8 MB)")

                st.markdown("**Step 2: 执行参数更新**")

                # 更新按钮
                if st.button("⚡ 开始微调 (Fine-tuning)", type="primary"):
                    progress_text = "任务初始化中..."
                    my_bar = st.progress(0, text=progress_text)

                    # --- 模拟训练过程 ---
                    steps = [
                        ("正在读取 CSV 数据...", 0.5),
                        ("数据清洗与归一化...", 1.0),
                        (f"加载用户 {st.session_state.username} 的私有权重...", 1.0),
                        ("启动反向传播 (Epoch 1/5)...", 1.5),
                        ("启动反向传播 (Epoch 5/5)...", 1.5),
                        ("验证集评估 (Accuracy: 98.2%)...", 1.0),
                        ("参数序列化与热部署...", 1.0)
                    ]

                    total_steps = len(steps)
                    for i, (msg, sleep_time) in enumerate(steps):
                        # 进度条逻辑
                        percent = int(((i) / total_steps) * 100)
                        my_bar.progress(percent, text=f"🔄 {msg}")
                        time.sleep(sleep_time)

                    my_bar.progress(100, text="✅ 更新完成")
                    st.balloons()
                    st.success(f"🎉 模型 `{selected_model}` 参数已更新至版本 V{random.randint(3, 9)}.0！")


def render_deep_model_logic(tool_name, tool_meta_name, context):
    """
    处理模型的训练、微调、私有化逻辑 (V4 最终完整版)
    核心机制：
    1. 只有当模型需要训练/微调且未完成时，返回 False (阻塞图表)。
    2. 当训练/微调完成后，返回 True (允许显示图表)，但通过不设置 ready_next 来暂停流程，等待用户点击按钮。
    """
    users = load_data(USER_DB_FILE)
    username = st.session_state.username
    user_models = users[username].get("model_states", {})

    tool_map = {
        "tool_trend_algo": "model_trend",
        "tool_risk_algo": "model_risk",
        "tool_water_algo": "model_water"
    }
    db_key = tool_map.get(tool_name, tool_name)
    current_status = user_models.get(db_key, "untrained")

    # 如果流程已结束（回看历史），直接放行渲染
    if st.session_state.get("workflow_finished"):
        return True

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"#### 🧠 模型控制台: {tool_meta_name}")
        with c2:
            if current_status == "untrained":
                st.caption("🔴 状态: 未训练 (公共)")
            else:
                st.caption("🟢 状态: 已就绪 (专属)")

        st.divider()

        # =================================================
        # 场景 A: 公共库工具 (首次训练)
        # =================================================
        if current_status == "untrained":
            # 1. 还没训练 -> 显示训练配置 -> 阻塞图表
            if not st.session_state.get(f"trained_{tool_name}"):
                st.info("检测到您是首次使用该模型，需要初始化训练参数。")

                st.markdown("##### Step 1: 导入训练数据集")
                uploaded_train = st.file_uploader("请上传历史生产数据 (CSV/Parquet)",
                                                  type=["csv", "parquet"],
                                                  key=f"up_train_{tool_name}")

                st.markdown("##### Step 2: 执行训练")
                btn_label = "⚡ 开始全量数据训练" if uploaded_train else "⚡ 使用默认样本集开始训练"

                if st.button(btn_label, key=f"btn_train_{tool_name}", use_container_width=True):
                    if uploaded_train:
                        st.toast(f"已加载数据: {uploaded_train.name}")

                    my_bar = st.progress(0, text="正在分配计算资源...")
                    for percent in range(100):
                        time.sleep(0.02)
                        my_bar.progress(percent + 1,
                                        text=f"Training Epoch {percent // 20}/5 | Loss: {random.uniform(0.1, 0.5):.4f}")
                    my_bar.empty()

                    # 标记训练完成，刷新页面
                    st.session_state[f"trained_{tool_name}"] = True
                    st.rerun()

                # 没训练完，不给看图，阻塞
                return False

            else:
                # 2. 训练已完成 -> 显示决策按钮 -> 【允许看图】
                st.success(f"✅ 训练完成 | 准确率: {random.uniform(94.0, 98.0):.1f}%")

                with st.container():
                    st.markdown("##### 🕵️ 结果评估与决策")
                    st.caption("下图为基于新训练权重的预测结果，请评估是否达标：")

                    col_a, col_b = st.columns(2)

                    # 只有点击了这里，流程才继续
                    if col_a.button("💾 效果不错，存入专属库", key=f"btn_yes_{tool_name}", type="primary",
                                    use_container_width=True):
                        users[username]["model_states"][db_key] = "private"
                        save_data(USER_DB_FILE, users)
                        st.toast("模型已保存至专属空间")
                        st.session_state[f"{tool_name}_ready_next"] = True
                        time.sleep(0.5)
                        st.rerun()

                    if col_b.button("➡️ 仅本次使用，继续", key=f"btn_no_{tool_name}", use_container_width=True):
                        st.toast("使用临时模型继续")
                        st.session_state[f"{tool_name}_ready_next"] = True
                        time.sleep(0.5)
                        st.rerun()

                # 【关键】返回 True，让工具脚本把图画出来给用户看
                return True

        # =================================================
        # 场景 B: 专属库工具 (已有模型)
        # =================================================
        elif current_status == "private":
            mode_key = f"{tool_name}_mode"

            # 1. 还没选模式 -> 阻塞
            if mode_key not in st.session_state:
                st.info("检测到您的专属模型。请选择运行模式：")
                b1, b2 = st.columns(2)
                if b1.button("🔄 要增量微调", key=f"ft_yes_{tool_name}", use_container_width=True):
                    st.session_state[mode_key] = "finetuning"
                    st.rerun()
                if b2.button("⏩ 不要增量微调 (直接使用)", key=f"ft_no_{tool_name}", use_container_width=True):
                    st.session_state[mode_key] = "direct"
                    st.rerun()
                return False

            # 2. 直接使用 -> 推理 -> 放行
            elif st.session_state[mode_key] == "direct":
                if not st.session_state.get(f"{tool_name}_simulated"):
                    with st.spinner("正在加载专属权重并执行推理..."):
                        time.sleep(1.5)
                    st.session_state[f"{tool_name}_simulated"] = True

                st.session_state[f"{tool_name}_ready_next"] = True
                return True

            # 3. 微调模式
            elif st.session_state[mode_key] == "finetuning":
                # A. 还没微调完 -> 阻塞
                if not st.session_state.get(f"{tool_name}_ft_done"):
                    st.markdown("##### 📤 上传增量校准数据")
                    ft_file = st.file_uploader("拖拽新数据到此处...", type=["csv"], key=f"ft_up_{tool_name}")

                    start_ft = st.button("🚀 启动增量训练 (Fine-tuning)", key=f"start_ft_{tool_name}", type="primary")

                    if start_ft:
                        if ft_file:
                            st.toast(f"收到增量数据: {ft_file.name}")
                        prog_bar = st.progress(0, text="启动增量训练...")
                        for i in range(100):
                            time.sleep(0.03)
                            prog_bar.progress(i + 1, text=f"Fine-tuning... | Loss: {random.uniform(0.01, 0.1):.4f}")
                        prog_bar.empty()

                        st.session_state[f"{tool_name}_ft_done"] = True
                        st.rerun()

                    # 还没微调完，不给看图
                    return False

                else:
                    # B. 微调已完成 -> 显示决策按钮 -> 【允许看图】
                    st.success(f"✅ 微调完成 | 新增样本: 128 | 准确率提升: +{random.uniform(0.5, 1.2):.2f}%")

                    with st.container():
                        st.markdown("##### 🕵️ 微调效果评估")
                        st.caption("下图是微调后的预测表现，请决定是否更新模型版本：")

                        btn1, btn2 = st.columns(2)

                        # 只有点击按钮，才放行下一步
                        if btn1.button("💾 保存并更新版本", key=f"save_ft_{tool_name}", type="primary",
                                       use_container_width=True):
                            st.toast(f"✅ 模型 {db_key} 版本已更新至 V{random.randint(4, 9)}.0")
                            st.session_state[f"{tool_name}_ready_next"] = True
                            time.sleep(1)
                            st.rerun()

                        if btn2.button("➡️ 效果一般，不保存", key=f"del_ft_{tool_name}", use_container_width=True):
                            st.toast("⚠️ 放弃微调参数，使用旧参数继续")
                            st.session_state[f"{tool_name}_ready_next"] = True
                            time.sleep(1)
                            st.rerun()

                    # 【核心】这里返回 True！
                    # 这意味着工具代码会继续执行，把预测图画在这些按钮的下方。
                    # 用户可以先看下面的图，再决定点上面的“保存”还是“不保存”。
                    return True

    return False


def render_analysis_page():
    st.title("💬 油气生产一体化智能系统")

    # 1. 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "您好！我是您的专属AI生产指挥官。请告诉我要分析的任务。"}
        ]

    # 2. 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if not msg.get("is_tool_process"):
                st.markdown(msg["content"])

    # 3. 聊天输入框
    if prompt := st.chat_input("请输入指令..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- 【关键修复】 彻底重置状态 ---
        # 确保新任务开始时，不会残留上一个任务的“已完成”标记
        st.session_state.workflow_step = 0
        st.session_state.workflow_finished = False
        st.session_state.current_workflow = None
        st.session_state.current_context = None

        # 清除所有工具相关的临时 flag，防止“抢跑”
        keys_to_clear = [
            k for k in st.session_state.keys()
            if k.endswith("_ready_next")
               or k.endswith("_ft_done")
               or k.endswith("_mode")
               or k.endswith("_simulated")
               or k.startswith("trained_")
               or k.endswith("_run_completed")
        ]
        for k in keys_to_clear:
            del st.session_state[k]
        # -------------------------------

        st.rerun()

    # 4. 核心：任务执行大框
    if st.session_state.get("current_workflow") or (
            st.session_state.messages and st.session_state.messages[-1]["role"] == "user"):

        # A. 初始化工作流 (仅在刚触发且还没生成workflow时)
        last_msg = st.session_state.messages[-1]
        if not st.session_state.get("current_workflow") and last_msg["role"] == "user":
            with st.chat_message("assistant"):
                # AI 思考过程模拟
                thinking_box = st.empty()
                thoughts = [
                    "🤔 正在解析自然语言指令...",
                    "🔍 检索知识库: 关联历史生产数据与地质特征...",
                    "🧠 激活推理引擎: 正在拆解任务步骤...",
                    "🔗 匹配算法模型: 识别到意图与 'Deep-Learning' 相关...",
                    "🛠️ 正在编排 Agent 工具链 (CoT)...",
                    "✨ 方案生成完毕，准备执行。"
                ]
                for thought in thoughts:
                    thinking_box.markdown(f"_{thought}_")
                    time.sleep(random.uniform(0.3, 0.8))
                thinking_box.empty()

                wf, ctx = plan_workflow(last_msg["content"])
                # 注入模型控制函数
                ctx['render_model_ui'] = render_deep_model_logic
                st.session_state.current_workflow = wf
                st.session_state.current_context = ctx

        # 获取状态数据
        workflow = st.session_state.get("current_workflow")
        context = st.session_state.get("current_context")
        current_step = st.session_state.get("workflow_step", 0)
        is_finished = st.session_state.get("workflow_finished", False)

        if workflow:
            # ================= [新增功能 1] 工作流可视化 =================
            with st.chat_message("assistant"):
                st.markdown(f"#### 🗺️ AI 任务执行路径规划")
                graph = graphviz.Digraph()
                graph.attr(rankdir='LR', size='10,4')
                graph.attr('node', shape='box', style='filled,rounded',
                           fontname='Microsoft YaHei', fillcolor='#e3f2fd', color='#2196f3')

                for idx, tool_id in enumerate(workflow):
                    meta = TOOL_META.get(tool_id, {"name": tool_id})
                    if idx < current_step:
                        fill, pen = '#c8e6c9', '#4caf50'  # 绿
                    elif idx == current_step and not is_finished:
                        fill, pen = '#fff9c4', '#fbc02d'  # 黄
                    else:
                        fill, pen = '#e3f2fd', '#2196f3'  # 蓝

                    node_label = f"{idx + 1}. {meta['name']}"
                    graph.node(str(idx), node_label, fillcolor=fill, color=pen)
                    if idx > 0:
                        graph.edge(str(idx - 1), str(idx), color='#b0bec5')

                st.graphviz_chart(graph, use_container_width=True)
                st.divider()

            # --- 定义内部渲染函数 ---
            def render_tool_steps():
                for i, tool_id in enumerate(workflow):
                    # 如果还没完成整个流程，且当前遍历到的步骤 > 当前实际步骤，停止渲染后续
                    if not is_finished and i > current_step:
                        break

                    meta = TOOL_META.get(tool_id, {"name": tool_id, "icon": "🔧"})
                    is_current_active = (i == current_step) and not is_finished

                    if is_finished:
                        step_title = f"Step {i + 1}: {meta['name']} (✅ 已完成)"
                        expander_open = False
                    else:
                        step_state = "🔄 执行中..." if is_current_active else "✅ 已完成"
                        step_title = f"Step {i + 1}: {meta['name']} ({step_state})"
                        expander_open = is_current_active

                    with st.expander(step_title, expanded=expander_open):
                        try:
                            module = importlib.import_module(f"tools.{tool_id}")

                            # 【优化】后台逻辑防抖：防止 UI 交互导致 run() 重复执行
                            # 只有当前步骤激活，且之前没跑过 run，才执行
                            step_run_key = f"step_{i}_run_completed"

                            if is_current_active:
                                if not st.session_state.get(step_run_key):
                                    delay_bar = st.progress(0, text=f"⏳ {meta['name']} 正在执行...")
                                    for k in range(100):
                                        time.sleep(0.01)
                                        delay_bar.progress(k + 1)
                                    delay_bar.empty()

                                    # 执行工具的后台逻辑
                                    module.run(context)
                                    # 标记该步 run 已跑完
                                    st.session_state[step_run_key] = True

                            # 渲染 UI 视图 (模型交互逻辑在这里触发)
                            if hasattr(module, 'view'):
                                module.view(context)

                            # 流程控制：决定何时跳到下一步
                            if is_current_active:
                                deep_models = ["tool_trend_algo", "tool_risk_algo", "tool_water_algo"]

                                # A. 如果是深度模型
                                if tool_id in deep_models:
                                    # 【关键修复】只有当 ready_next 标志位被逻辑函数置为 True 时，才跳转
                                    if st.session_state.get(f"{tool_id}_ready_next"):
                                        time.sleep(0.5)
                                        st.session_state.workflow_step += 1
                                        st.rerun()
                                    # 否则这里什么都不做，静静等待用户操作

                                # B. 如果是数据加载 (Data Loader)
                                elif tool_id == "tool_data_loader":
                                    st.write("---")
                                    if st.button("⬇️ 数据确认无误，执行下一步", key=f"next_step_{i}", type="primary"):
                                        st.session_state.workflow_step += 1
                                        st.rerun()

                                # C. 普通工具 (Cleaner, Feature, etc.)
                                else:
                                    time.sleep(0.8)  # 简单展示后自动跳转
                                    st.session_state.workflow_step += 1
                                    st.rerun()

                        except Exception as e:
                            st.error(f"执行出错: {e}")

            # --- 核心容器切换逻辑 ---
            if is_finished:
                with st.expander("✅ 所有步骤执行完毕 (点击查看详情/操作历史)"):
                    render_tool_steps()
            else:
                status_label = f"🚀 正在执行: {context.get('task_name')} (Step {current_step + 1}/{len(workflow)})"
                with st.status(status_label, expanded=True) as status:
                    render_tool_steps()

            # 判断结束
            if not is_finished and current_step >= len(workflow):
                st.session_state.workflow_finished = True
                st.rerun()

    # 5. 任务完成后生成最终总结
    if st.session_state.get("workflow_finished", False):
        last_msg = st.session_state.messages[-1]
        if last_msg["role"] == "user" or (last_msg["role"] == "assistant" and "核心结论" not in last_msg["content"]):
            ctx = st.session_state.current_context
            summary = ctx.get('trend_summary') or ctx.get('risk_summary') or ctx.get('water_summary') or "分析完成。"
            final_resp = f"**{ctx.get('task_name')}** 执行完成。\n\n📊 **核心结论**: {summary}\n\n详细过程请查看上方折叠面板。"

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_resp,
                "is_tool_process": False
            })
            st.rerun()


if __name__ == "__main__":
    init_db()  # 初始化文件
    init_session()

    if not st.session_state.logged_in:
        render_login_page()
    else:
        render_sidebar()  # 侧边栏常驻

        # 根据角色和页面路由
        if st.session_state.role == 'admin':
            render_manager_page()
        else:
            if st.session_state.current_page == "analysis":
                render_analysis_page()
            elif st.session_state.current_page == "training":
                render_training_page()
