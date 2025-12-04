import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.integrate as spi
import streamlit as st
import time
from plotly.subplots import make_subplots
from scipy import stats

# 页面设置
st.set_page_config(
    page_title="蒙特卡洛模拟：从随机性到建模洞察",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS美化页面
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2ca02c;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .highlight-box {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border-left: 5px solid #1f77b4;
    }
    .formula {
        font-family: "Times New Roman", serif;
        font-size: 1.2rem;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.title("🎲 蒙特卡洛模拟教学")
    st.markdown("---")

    # 案例选择
    st.subheader("案例选择")
    selected_case = st.radio(
        "选择要探索的案例:",
        ["引言与理论", "案例1: 估算圆周率π", "案例2: 计算定积分", "案例3: 排队系统", "例题练习", "Q&A环节"]
    )

    st.markdown("---")

    # 全局设置
    st.subheader("全局设置")
    seed = st.number_input("随机数种子", value=42, min_value=0, help="固定随机种子使结果可重现")
    np.random.seed(seed)

    st.markdown("---")
    st.caption("开发: 蒙特卡洛方法教学团队")
    st.caption("适用对象: 具有基础统计学知识的学生")

# 主页面标题
st.markdown('<h1 class="main-header">蒙特卡洛模拟：从随机性到建模洞察</h1>', unsafe_allow_html=True)

# ============================================
# 第一部分：引言与理论
# ============================================
if selected_case == "引言与理论":
    st.markdown('<h2 class="section-header">用随机数解决确定性问题</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **蒙特卡洛方法**是一种利用随机抽样来解决确定性数学问题的数值计算方法。

        ### 核心矛盾点
        > 如何用"随机性"来解决"确定性"问题？

        看似矛盾，实则体现了概率论和大数定律的强大力量：
        - 某些确定性问题难以用解析方法求解
        - 但可以通过构造概率模型，用随机抽样获得近似解
        - 随着抽样次数增加，近似解收敛于真实值

        ### 历史起源
        蒙特卡洛方法在**曼哈顿计划**中首次系统性地用于核武器研制：
        - 得名于摩纳哥的蒙特卡洛赌城（随机性）
        - 冯·诺依曼、乌拉姆等人为解决中子扩散问题而发展
        - 开启了随机模拟在科学计算中的新时代
        """)

    with col2:
        # 显示一个简单的蒙特卡洛模拟示意图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[0, 0, 1, 1, 0],
            mode='lines',
            name='正方形',
            line=dict(color='blue', width=2)
        ))

        # 生成圆的坐标
        theta = np.linspace(0, np.pi / 2, 100)
        x_circle = 0.5 + 0.5 * np.cos(theta)
        y_circle = 0.5 + 0.5 * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            name='四分之一圆',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title="蒙特卡洛方法示意图",
            xaxis=dict(range=[0, 1], constrain="domain"),
            yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
            width=400,
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 蒙特卡洛方法的优势
    - **高维问题**：维度增加时，传统数值方法计算量指数增长，蒙特卡洛方法受影响较小
    - **复杂边界**：可处理复杂几何形状和边界条件
    - **并行性**：随机样本相互独立，易于并行计算
    - **灵活性**：适用于各类数学问题（积分、优化、模拟等）
    """)

    st.markdown("---")
    st.markdown('<h2 class="section-header">蒙特卡洛方法的四步法</h2>', unsafe_allow_html=True)

    # 使用列展示四步法
    steps = st.columns(4)

    with steps[0]:
        st.info("**1. 建模**")
        st.markdown("将实际问题转化为概率模型")
        st.latex(r"P(\text{问题解}) = \mathbb{E}[g(X)]")

    with steps[1]:
        st.success("**2. 抽样**")
        st.markdown("从概率分布生成随机样本")
        st.latex(r"X_1, X_2, \dots, X_N \sim p(x)")

    with steps[2]:
        st.warning("**3. 模拟**")
        st.markdown("对样本执行确定性计算")
        st.latex(r"Y_i = g(X_i)")

    with steps[3]:
        st.error("**4. 估计**")
        st.markdown("利用大数定律计算估计值")
        st.latex(r"\hat{I}_N = \frac{1}{N} \sum_{i=1}^N Y_i")

    st.markdown("---")
    st.markdown('<h2 class="section-header">关键数学推导：蒙特卡洛积分</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 从定积分到期望估计

    考虑定积分问题：
    """)

    st.latex(r"I = \int_a^b f(x) dx")

    st.markdown("""
    设随机变量 $X$ 在 $[a, b]$ 上服从均匀分布：
    """)

    st.latex(r"X \sim U(a, b), \quad p(x) = \frac{1}{b-a}")

    st.markdown("则函数 $f(X)$ 的数学期望为：")

    st.latex(r"\mathbb{E}[f(X)] = \int_a^b f(x) p(x) dx = \frac{1}{b-a} \int_a^b f(x) dx")

    st.markdown("由此可得：")

    st.latex(r"\int_a^b f(x) dx = (b-a) \cdot \mathbb{E}[f(X)]")

    st.markdown("""
    根据大数定律，样本均值收敛于期望值：
    """)

    st.latex(r"\hat{I}_N = (b-a) \cdot \frac{1}{N} \sum_{i=1}^N f(X_i) \xrightarrow[N\to\infty]{} I")

    with st.expander("📚 推导细节与解释"):
        st.markdown("""
        **推导过程详解**：

        1. **定义均匀分布随机变量**：$X \\sim U(a, b)$，其概率密度函数为 $p(x) = \\frac{1}{b-a}$

        2. **计算 $f(X)$ 的期望**：
           $$
           \\mathbb{E}[f(X)] = \\int_a^b f(x) p(x) dx = \\int_a^b f(x) \\cdot \\frac{1}{b-a} dx
           $$

        3. **整理得到**：
           $$
           \\int_a^b f(x) dx = (b-a) \\cdot \\mathbb{E}[f(X)]
           $$

        4. **由大数定律，用样本均值估计期望**：
           $$
           \\hat{I}_N = (b-a) \\cdot \\frac{1}{N} \\sum_{i=1}^N f(X_i)
           $$

        5. **估计误差**：根据中心极限定理，估计量近似服从正态分布：
           $$
           \\hat{I}_N \\sim \\mathcal{N}\\left(I, \\frac{(b-a)^2 \\sigma_f^2}{N}\\right)
           $$
           其中 $\\sigma_f^2 = \\text{Var}[f(X)]$

        **关键洞察**：
        - 将确定性的积分问题转化为随机变量的期望估计问题
        - 利用大数定律保证估计的一致性
        - 误差以 $O(1/\\sqrt{N})$ 的速度收敛，与维度无关
        """)

# ============================================
# 第二部分：案例1 - 估算圆周率π
# ============================================
elif selected_case == "案例1: 估算圆周率π":
    st.markdown('<h2 class="section-header">案例1：估算圆周率π（几何建模）</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 数学模型：正方形与内切圆

    考虑单位正方形 $[0,1] \\times [0,1]$ 及其内切四分之一圆：
    - 正方形面积：$A_{\\text{square}} = 1$
    - 四分之一圆面积：$A_{\\text{quarter circle}} = \\frac{\\pi}{4}$
    - 面积比：$\\frac{A_{\\text{quarter circle}}}{A_{\\text{square}}} = \\frac{\\pi}{4}$

    因此：
    """)

    st.latex(r"\pi = 4 \times \frac{\text{落在四分之一圆内的点数}}{\text{总点数}}")

    # 控制参数
    st.subheader("模拟参数设置")
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.slider("模拟次数 N", min_value=100, max_value=10000, value=1000, step=100)

    with col2:
        point_size = st.slider("点的大小", min_value=1, max_value=10, value=3)

    with col3:
        show_animation = st.checkbox("显示动画效果", value=True)


    # 模拟函数
    def estimate_pi_monte_carlo(N, animate=False):
        """使用蒙特卡洛方法估计π值"""
        # 步骤1: 建模 - 生成均匀分布的随机点
        x = np.random.uniform(0, 1, N)
        y = np.random.uniform(0, 1, N)

        # 步骤2: 抽样 - 已通过随机数生成实现

        # 步骤3: 模拟 - 判断点是否在圆内
        distances = x ** 2 + y ** 2
        inside_circle = distances <= 1

        # 步骤4: 估计 - 计算π的估计值
        pi_estimate = 4 * np.sum(inside_circle) / N

        # 计算误差和置信区间
        true_pi = np.pi
        error = abs(pi_estimate - true_pi)

        # 计算95%置信区间
        p_hat = np.sum(inside_circle) / N
        se = 4 * np.sqrt(p_hat * (1 - p_hat) / N)  # 标准误
        ci_lower = pi_estimate - 1.96 * se
        ci_upper = pi_estimate + 1.96 * se

        return x, y, inside_circle, pi_estimate, error, ci_lower, ci_upper


    # 运行模拟
    x, y, inside_circle, pi_estimate, error, ci_lower, ci_upper = estimate_pi_monte_carlo(N)

    # 结果显示
    col1, col2 = st.columns(2)

    with col1:
        st.metric("π的估计值", f"{pi_estimate:.6f}")
        st.metric("绝对误差", f"{error:.6f}")
        st.metric("相对误差", f"{100 * error / np.pi:.4f}%")

    with col2:
        st.metric("95%置信区间", f"[{ci_lower:.6f}, {ci_upper:.6f}]")
        st.metric("区间宽度", f"{ci_upper - ci_lower:.6f}")
        st.metric("落在圆内的点数", f"{np.sum(inside_circle)} / {N}")

    # 可视化
    st.subheader("可视化：随机点分布")

    # 创建交互式图表
    fig = go.Figure()

    # 添加正方形边界
    fig.add_trace(go.Scatter(
        x=[0, 1, 1, 0, 0],
        y=[0, 0, 1, 1, 0],
        mode='lines',
        name='正方形',
        line=dict(color='blue', width=2)
    ))

    # 添加四分之一圆边界
    theta = np.linspace(0, np.pi / 2, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        name='四分之一圆',
        line=dict(color='red', width=2)
    ))

    # 添加随机点（区分圆内和圆外）
    fig.add_trace(go.Scatter(
        x=x[inside_circle],
        y=y[inside_circle],
        mode='markers',
        name='圆内点',
        marker=dict(size=point_size, color='green', opacity=0.7)
    ))

    fig.add_trace(go.Scatter(
        x=x[~inside_circle],
        y=y[~inside_circle],
        mode='markers',
        name='圆外点',
        marker=dict(size=point_size, color='orange', opacity=0.7)
    ))

    fig.update_layout(
        title=f"蒙特卡洛法估算π (N={N})",
        xaxis=dict(title='x', range=[0, 1], scaleanchor="y", scaleratio=1),
        yaxis=dict(title='y', range=[0, 1]),
        width=700,
        height=600,
        showlegend=True,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 收敛性分析
    st.subheader("收敛性分析")

    # 分析不同N值下的估计精度
    N_values = np.logspace(2, 4, 20).astype(int)
    pi_estimates = []
    errors = []

    for n in N_values:
        _, _, _, pi_est, err, _, _ = estimate_pi_monte_carlo(n, animate=False)
        pi_estimates.append(pi_est)
        errors.append(err)

    # 创建收敛图
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("π估计值随N的变化", "误差随N的变化"))

    fig2.add_trace(
        go.Scatter(x=N_values, y=pi_estimates, mode='lines+markers', name='估计值'),
        row=1, col=1
    )
    fig2.add_hline(y=np.pi, line_dash="dash", line_color="red", name="真实值", row=1, col=1)

    fig2.add_trace(
        go.Scatter(x=N_values, y=errors, mode='lines+markers', name='绝对误差'),
        row=1, col=2
    )
    fig2.add_trace(
        go.Scatter(x=N_values, y=1 / np.sqrt(N_values), mode='lines', name='1/√N', line=dict(dash='dash')),
        row=1, col=2
    )

    fig2.update_xaxes(title_text="模拟次数 N", type="log", row=1, col=1)
    fig2.update_xaxes(title_text="模拟次数 N", type="log", row=1, col=2)
    fig2.update_yaxes(title_text="π估计值", row=1, col=1)
    fig2.update_yaxes(title_text="绝对误差", type="log", row=1, col=2)

    fig2.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

    # 代码展示
    st.subheader("实现代码")

    code = '''
def estimate_pi_monte_carlo(N):
    """使用蒙特卡洛方法估计π值"""
    # 步骤1: 建模 - 生成均匀分布的随机点
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    # 步骤2: 抽样 - 已通过随机数生成实现

    # 步骤3: 模拟 - 判断点是否在圆内
    distances = x**2 + y**2
    inside_circle = distances <= 1  # 点在圆内的条件

    # 步骤4: 估计 - 计算π的估计值
    pi_estimate = 4 * np.sum(inside_circle) / N

    # 计算95%置信区间
    p_hat = np.sum(inside_circle) / N
    se = 4 * np.sqrt(p_hat * (1 - p_hat) / N)  # 标准误
    ci_lower = pi_estimate - 1.96 * se
    ci_upper = pi_estimate + 1.96 * se

    return pi_estimate, ci_lower, ci_upper, inside_circle
    '''

    st.code(code, language='python')

# ============================================
# 第三部分：案例2 - 计算定积分
# ============================================
elif selected_case == "案例2: 计算定积分":
    st.markdown('<h2 class="section-header">案例2：计算复杂定积分（函数建模）</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 蒙特卡洛积分法

    根据之前的推导，定积分可以转化为期望估计问题：
    """)

    st.latex(r"\int_a^b f(x) dx = (b-a) \cdot \mathbb{E}[f(X)], \quad X \sim U(a, b)")

    st.markdown("### 选择要计算的积分")

    # 积分选择
    integral_option = st.selectbox(
        "选择积分问题:",
        [
            "示例1: ∫₀¹ e^{-x²} dx (高斯积分)",
            "示例2: ∫₀^{2π} sin(x²) dx (震荡函数)",
            "自定义积分"
        ]
    )

    # 根据选择设置积分参数
    if integral_option == "示例1: ∫₀¹ e^{-x²} dx (高斯积分)":
        a, b = 0, 1


        def f(x):
            return np.exp(-x ** 2)


        func_str = "e^{-x²}"
        true_value = 0.746824132812427  # 已知近似值

    elif integral_option == "示例2: ∫₀^{2π} sin(x²) dx (震荡函数)":
        a, b = 0, 2 * np.pi


        def f(x):
            return np.sin(x ** 2)


        func_str = "sin(x²)"
        true_value = 0.601  # 近似值

    else:  # 自定义积分
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("积分下限 a", value=0.0, step=0.1)
        with col2:
            b = st.number_input("积分上限 b", value=1.0, step=0.1)

        # 自定义函数输入
        func_input = st.text_input("输入被积函数 f(x) (使用numpy语法)", "np.exp(-x**2)")
        func_str = func_input.replace("np.", "")


        # 定义函数
        def f(x):
            try:
                # 安全地评估函数
                return eval(func_input, {"np": np, "x": x})
            except:
                st.error("函数表达式错误，请使用有效的numpy语法")
                return np.exp(-x ** 2)


        # 计算参考值
        try:
            true_value, _ = spi.quad(f, a, b)
        except:
            true_value = None

    # 模拟参数
    st.subheader("模拟参数设置")
    N_integral = st.slider("模拟次数 N", min_value=100, max_value=50000, value=5000, step=100)


    # 运行蒙特卡洛积分
    def monte_carlo_integral(f, a, b, N):
        """蒙特卡洛积分"""
        # 生成均匀分布的随机样本
        x_samples = np.random.uniform(a, b, N)

        # 计算函数值
        f_values = f(x_samples)

        # 计算积分估计
        integral_estimate = (b - a) * np.mean(f_values)

        # 计算标准差和置信区间
        f_std = np.std(f_values)
        se = (b - a) * f_std / np.sqrt(N)  # 标准误
        ci_lower = integral_estimate - 1.96 * se
        ci_upper = integral_estimate + 1.96 * se

        return integral_estimate, ci_lower, ci_upper, f_std, se, x_samples, f_values


    # 运行模拟
    integral_estimate, ci_lower, ci_upper, f_std, se, x_samples, f_values = monte_carlo_integral(f, a, b, N_integral)

    # 结果显示
    st.subheader("结果对比")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("蒙特卡洛估计", f"{integral_estimate:.8f}")

    with col2:
        if true_value is not None:
            error = abs(integral_estimate - true_value)
            st.metric("绝对误差", f"{error:.8f}")

    with col3:
        if true_value is not None:
            rel_error = 100 * error / abs(true_value)
            st.metric("相对误差", f"{rel_error:.4f}%")

    col4, col5 = st.columns(2)

    with col4:
        st.metric("95%置信区间", f"[{ci_lower:.8f}, {ci_upper:.8f}]")

    with col5:
        st.metric("区间宽度", f"{ci_upper - ci_lower:.8f}")

    # 可视化
    st.subheader("可视化：函数曲线与随机采样")

    # 创建函数曲线
    x_curve = np.linspace(a, b, 1000)
    y_curve = f(x_curve)

    fig = go.Figure()

    # 添加函数曲线
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        name=f'f(x) = {func_str}',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)'
    ))

    # 添加随机样本点
    fig.add_trace(go.Scatter(
        x=x_samples[:200],  # 只显示部分点以免过于密集
        y=f_values[:200],
        mode='markers',
        name='随机样本',
        marker=dict(size=5, color='red', opacity=0.6)
    ))

    fig.update_layout(
        title=f"蒙特卡洛积分: ∫_{{{a}}}^{{{b}}} {func_str} dx",
        xaxis_title='x',
        yaxis_title='f(x)',
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # 与scipy数值积分对比
    st.subheader("与数值积分方法对比")

    if true_value is not None:
        # 使用scipy进行数值积分
        scipy_result, scipy_error = spi.quad(f, a, b)

        # 创建对比数据
        methods = ['蒙特卡洛', 'Scipy (quad)']
        estimates = [integral_estimate, scipy_result]
        errors_to_true = [abs(integral_estimate - true_value), abs(scipy_result - true_value)]

        fig_compare = make_subplots(rows=1, cols=2, subplot_titles=("积分估计值对比", "绝对误差对比"))

        fig_compare.add_trace(
            go.Bar(x=methods, y=estimates, name='估计值', marker_color=['blue', 'green']),
            row=1, col=1
        )
        fig_compare.add_hline(y=true_value, line_dash="dash", line_color="red", name="参考值", row=1, col=1)

        fig_compare.add_trace(
            go.Bar(x=methods, y=errors_to_true, name='绝对误差', marker_color=['orange', 'red']),
            row=1, col=2
        )

        fig_compare.update_xaxes(title_text="方法", row=1, col=1)
        fig_compare.update_xaxes(title_text="方法", row=1, col=2)
        fig_compare.update_yaxes(title_text="积分值", row=1, col=1)
        fig_compare.update_yaxes(title_text="绝对误差", row=1, col=2)

        fig_compare.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_compare, use_container_width=True)

    # 收敛性分析
    st.subheader("收敛性分析")

    # 测试不同N值下的表现
    N_test_values = np.logspace(2, 4.5, 20).astype(int)
    mc_estimates = []
    mc_errors = []

    for n in N_test_values:
        est, _, _, _, _, _, _ = monte_carlo_integral(f, a, b, n)
        mc_estimates.append(est)
        if true_value is not None:
            mc_errors.append(abs(est - true_value))

    # 创建收敛图
    if true_value is not None:
        fig_convergence = go.Figure()

        fig_convergence.add_trace(go.Scatter(
            x=N_test_values,
            y=mc_estimates,
            mode='lines+markers',
            name='蒙特卡洛估计',
            line=dict(color='blue')
        ))

        fig_convergence.add_trace(go.Scatter(
            x=N_test_values,
            y=mc_errors,
            mode='lines+markers',
            name='绝对误差',
            line=dict(color='red'),
            yaxis='y2'
        ))

        fig_convergence.add_trace(go.Scatter(
            x=N_test_values,
            y=1 / np.sqrt(N_test_values),
            mode='lines',
            name='1/√N 参考线',
            line=dict(color='green', dash='dash'),
            yaxis='y2'
        ))

        fig_convergence.update_layout(
            title="蒙特卡洛积分收敛性",
            xaxis=dict(title="模拟次数 N", type="log"),
            yaxis=dict(title="积分估计值"),
            yaxis2=dict(title="绝对误差", overlaying="y", side="right", type="log"),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_convergence, use_container_width=True)

    # 代码展示
    st.subheader("实现代码")

    code = '''
def monte_carlo_integral(f, a, b, N):
    """蒙特卡洛积分"""
    # 步骤1: 建模 - 将积分转化为期望问题
    # I = ∫_a^b f(x) dx = (b-a) * E[f(X)], X~U(a,b)

    # 步骤2: 抽样 - 生成均匀分布的随机样本
    x_samples = np.random.uniform(a, b, N)

    # 步骤3: 模拟 - 计算函数值
    f_values = f(x_samples)

    # 步骤4: 估计 - 计算样本均值作为期望估计
    integral_estimate = (b - a) * np.mean(f_values)

    # 误差估计 - 计算标准误和置信区间
    f_std = np.std(f_values, ddof=1)  # 样本标准差
    se = (b - a) * f_std / np.sqrt(N)  # 标准误
    ci_lower = integral_estimate - 1.96 * se  # 95%置信下限
    ci_upper = integral_estimate + 1.96 * se  # 95%置信上限

    return integral_estimate, ci_lower, ci_upper
    '''

    st.code(code, language='python')

# ============================================
# 第四部分：案例3 - 排队系统
# ============================================
elif selected_case == "案例3: 排队系统":
    st.markdown('<h2 class="section-header">案例3：单服务台排队系统（动态系统建模）</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### M/M/1 排队系统模型

    **模型假设**：
    - 顾客到达过程：泊松过程，到达间隔服从指数分布，参数λ（到达率）
    - 服务时间：服从指数分布，参数μ（服务率）
    - 单服务台，先到先服务，无限队列容量

    **关键性能指标**：
    - 平均队长：系统中平均顾客数
    - 平均等待时间：顾客在队列中的平均等待时间
    - 服务台利用率：ρ = λ/μ (要求 ρ < 1 系统稳定)
    """)

    # 理论公式
    with st.expander("📈 M/M/1排队系统的理论公式"):
        st.markdown("""
        **理论公式（稳态条件下）**：

        - 服务台利用率：$\\rho = \\frac{\\lambda}{\\mu}$
        - 平均队长：$L = \\frac{\\rho}{1-\\rho}$
        - 平均等待时间：$W = \\frac{L}{\\lambda} = \\frac{1}{\\mu - \\lambda}$
        - 系统中顾客数为n的概率：$P_n = (1-\\rho)\\rho^n$

        **稳定性条件**：$\\lambda < \\mu$ (到达率小于服务率)
        """)

    # 模拟参数设置
    st.subheader("系统参数设置")

    col1, col2, col3 = st.columns(3)

    with col1:
        lam = st.slider("到达率 λ (顾客/分钟)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

    with col2:
        mu = st.slider("服务率 μ (顾客/分钟)", min_value=0.2, max_value=6.0, value=1.0, step=0.1)

    with col3:
        sim_time = st.slider("模拟时间 (分钟)", min_value=100, max_value=5000, value=1000, step=100)

    # 检查稳定性
    rho = lam / mu
    if rho >= 1:
        st.error(f"⚠️ 系统不稳定！ρ = {rho:.2f} ≥ 1。请确保 λ < μ。")
        st.stop()

    # 理论计算
    theoretical_queue_length = rho / (1 - rho)
    theoretical_wait_time = 1 / (mu - lam)


    # 蒙特卡洛模拟函数
    def simulate_mm1_queue(lam, mu, sim_time):
        """模拟M/M/1排队系统"""
        time = 0
        queue_length = 0
        server_busy = False
        next_arrival = np.random.exponential(1 / lam)
        next_departure = float('inf')

        # 记录统计数据
        queue_lengths = [0]
        event_times = [0]
        total_customers = 0
        total_wait_time = 0
        total_service_time = 0

        # 模拟事件
        while time < sim_time:
            # 判断下一个事件是到达还是离开
            if next_arrival < next_departure:
                # 到达事件
                time = next_arrival
                event_times.append(time)
                total_customers += 1

                # 如果服务台空闲，立即开始服务
                if not server_busy:
                    server_busy = True
                    service_time = np.random.exponential(1 / mu)
                    next_departure = time + service_time
                    total_service_time += service_time
                else:
                    # 否则加入队列
                    queue_length += 1

                # 安排下一个到达
                next_arrival = time + np.random.exponential(1 / lam)

            else:
                # 离开事件
                time = next_departure
                event_times.append(time)

                # 如果队列中有顾客，下一个顾客开始服务
                if queue_length > 0:
                    queue_length -= 1
                    service_time = np.random.exponential(1 / mu)
                    next_departure = time + service_time
                    total_service_time += service_time
                    # 累积等待时间
                    total_wait_time += service_time
                else:
                    # 队列为空，服务台空闲
                    server_busy = False
                    next_departure = float('inf')

            queue_lengths.append(queue_length)

        # 计算平均队长和平均等待时间
        # 使用时间加权平均计算平均队长
        avg_queue_length = 0
        for i in range(len(queue_lengths) - 1):
            duration = event_times[i + 1] - event_times[i]
            avg_queue_length += queue_lengths[i] * duration

        avg_queue_length /= event_times[-1] if event_times[-1] > 0 else 1

        # 平均等待时间（Little's Law: L = λW）
        if total_customers > 0:
            avg_wait_time = avg_queue_length / lam
        else:
            avg_wait_time = 0

        return avg_queue_length, avg_wait_time, total_customers, queue_lengths, event_times


    # 运行模拟
    avg_queue_length_sim, avg_wait_time_sim, total_customers, queue_lengths, event_times = simulate_mm1_queue(lam, mu,
                                                                                                              sim_time)

    # 结果显示
    st.subheader("模拟结果与理论对比")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("平均队长 (模拟)", f"{avg_queue_length_sim:.3f}")
        st.metric("平均队长 (理论)", f"{theoretical_queue_length:.3f}")
        diff_queue = abs(avg_queue_length_sim - theoretical_queue_length)
        st.metric("差异", f"{diff_queue:.3f}")

    with col2:
        st.metric("平均等待时间 (模拟)", f"{avg_wait_time_sim:.3f} 分钟")
        st.metric("平均等待时间 (理论)", f"{theoretical_wait_time:.3f} 分钟")
        diff_wait = abs(avg_wait_time_sim - theoretical_wait_time)
        st.metric("差异", f"{diff_wait:.3f} 分钟")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("服务台利用率 ρ", f"{rho:.3f}")
        st.metric("总模拟时间", f"{sim_time:.0f} 分钟")

    with col4:
        st.metric("服务顾客总数", f"{total_customers}")
        st.metric("稳定性", "稳定" if rho < 1 else "不稳定",
                  delta="ρ < 1" if rho < 1 else "ρ ≥ 1")

    # 可视化队列动态
    st.subheader("队列长度随时间变化")

    # 创建队列长度随时间变化的图表
    fig = go.Figure()

    # 由于事件是离散的，我们需要创建一个连续的时间序列来显示队列长度
    # 这里我们简单地将事件之间的队列长度视为常数
    time_points = []
    queue_points = []

    for i in range(len(event_times) - 1):
        time_points.append(event_times[i])
        time_points.append(event_times[i + 1])
        queue_points.append(queue_lengths[i])
        queue_points.append(queue_lengths[i])

    # 只显示前200个事件以免图表过于密集
    max_points = min(400, len(time_points))

    fig.add_trace(go.Scatter(
        x=time_points[:max_points],
        y=queue_points[:max_points],
        mode='lines',
        name='队列长度',
        line=dict(color='blue', width=1)
    ))

    # 添加平均队长水平线
    fig.add_hline(y=avg_queue_length_sim, line_dash="dash", line_color="red",
                  name=f"平均队长: {avg_queue_length_sim:.2f}")

    fig.add_hline(y=theoretical_queue_length, line_dash="dash", line_color="green",
                  name=f"理论队长: {theoretical_queue_length:.2f}")

    fig.update_layout(
        title="队列长度随时间变化",
        xaxis_title="时间 (分钟)",
        yaxis_title="队列长度",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # 性能指标随ρ的变化
    st.subheader("系统性能随利用率ρ的变化")

    # 计算不同ρ值下的理论性能
    rho_values = np.linspace(0.1, 0.95, 20)
    theoretical_lengths = rho_values / (1 - rho_values)
    theoretical_waits = 1 / (mu * (1 - rho_values))

    # 运行模拟获取实际值
    simulated_lengths = []
    simulated_waits = []

    for r in rho_values:
        lam_test = r * mu
        if lam_test < mu:  # 确保稳定性
            avg_len, avg_wait, _, _, _ = simulate_mm1_queue(lam_test, mu, 500)
            simulated_lengths.append(avg_len)
            simulated_waits.append(avg_wait)
        else:
            simulated_lengths.append(np.nan)
            simulated_waits.append(np.nan)

    # 创建对比图表
    fig_rho = make_subplots(rows=1, cols=2, subplot_titles=("平均队长 vs 利用率ρ", "平均等待时间 vs 利用率ρ"))

    fig_rho.add_trace(
        go.Scatter(x=rho_values, y=theoretical_lengths, mode='lines', name='理论值', line=dict(color='blue')),
        row=1, col=1
    )
    fig_rho.add_trace(
        go.Scatter(x=rho_values, y=simulated_lengths, mode='markers', name='模拟值', marker=dict(color='red', size=8)),
        row=1, col=1
    )

    fig_rho.add_trace(
        go.Scatter(x=rho_values, y=theoretical_waits, mode='lines', name='理论值', line=dict(color='blue')),
        row=1, col=2
    )
    fig_rho.add_trace(
        go.Scatter(x=rho_values, y=simulated_waits, mode='markers', name='模拟值', marker=dict(color='red', size=8)),
        row=1, col=2
    )

    fig_rho.update_xaxes(title_text="利用率 ρ", row=1, col=1)
    fig_rho.update_xaxes(title_text="利用率 ρ", row=1, col=2)
    fig_rho.update_yaxes(title_text="平均队长", row=1, col=1)
    fig_rho.update_yaxes(title_text="平均等待时间", row=1, col=2)

    fig_rho.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_rho, use_container_width=True)

    # 代码展示
    st.subheader("实现代码")

    code = '''
def simulate_mm1_queue(lam, mu, sim_time):
    """模拟M/M/1排队系统"""
    time = 0
    queue_length = 0
    server_busy = False
    next_arrival = np.random.exponential(1/lam)  # 第一个到达时间
    next_departure = float('inf')  # 初始时没有离开事件

    # 记录统计数据
    queue_lengths = [0]
    event_times = [0]
    total_customers = 0

    # 模拟循环
    while time < sim_time:
        # 判断下一个事件类型
        if next_arrival < next_departure:
            # 到达事件
            time = next_arrival
            total_customers += 1

            if not server_busy:
                # 服务台空闲，立即开始服务
                server_busy = True
                service_time = np.random.exponential(1/mu)
                next_departure = time + service_time
            else:
                # 服务台忙碌，加入队列
                queue_length += 1

            # 安排下一个到达
            next_arrival = time + np.random.exponential(1/lam)
        else:
            # 离开事件
            time = next_departure

            if queue_length > 0:
                # 队列中有顾客等待，下一个开始服务
                queue_length -= 1
                service_time = np.random.exponential(1/mu)
                next_departure = time + service_time
            else:
                # 队列为空，服务台变为空闲
                server_busy = False
                next_departure = float('inf')

        # 记录状态
        queue_lengths.append(queue_length)
        event_times.append(time)

    # 计算时间加权平均队长
    avg_queue_length = 0
    for i in range(len(queue_lengths)-1):
        duration = event_times[i+1] - event_times[i]
        avg_queue_length += queue_lengths[i] * duration
    avg_queue_length /= event_times[-1]

    # 根据Little定律计算平均等待时间
    avg_wait_time = avg_queue_length / lam

    return avg_queue_length, avg_wait_time, total_customers
    '''

    st.code(code, language='python')

# ============================================
# 第五部分：例题练习
# ============================================
elif selected_case == "例题练习":
    st.markdown('<h2 class="section-header">例题练习：自定义蒙特卡洛积分</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 练习：使用蒙特卡洛方法计算定积分

    在这个练习中，你可以自定义积分问题，并使用蒙特卡洛方法进行计算。
    尝试不同的函数和积分区间，观察蒙特卡洛方法的精度和收敛性。
    """)

    # 自定义积分参数
    st.subheader("设置积分参数")

    col1, col2 = st.columns(2)

    with col1:
        a_ex = st.number_input("积分下限 a", value=0.0, step=0.1, key="ex_a")

    with col2:
        b_ex = st.number_input("积分上限 b", value=1.0, step=0.1, key="ex_b")

    # 函数选择
    st.subheader("选择或自定义被积函数")

    func_options = {
        "f(x) = x²": "x**2",
        "f(x) = sin(x)": "np.sin(x)",
        "f(x) = e^(-x)": "np.exp(-x)",
        "f(x) = cos(x²)": "np.cos(x**2)",
        "f(x) = 1/(1+x²)": "1/(1+x**2)",
        "自定义函数": "custom"
    }

    selected_func_name = st.selectbox("选择函数", list(func_options.keys()))

    if selected_func_name == "自定义函数":
        custom_func = st.text_input("输入自定义函数 f(x) (使用numpy语法)", "np.sin(x) * np.exp(-x)")
        func_expr = custom_func
    else:
        func_expr = func_options[selected_func_name]


    # 定义被积函数
    def f_ex(x):
        try:
            return eval(func_expr, {"np": np, "x": x})
        except:
            st.error("函数表达式错误，请使用有效的numpy语法")
            return np.sin(x)


    # 模拟参数
    st.subheader("模拟参数")
    N_ex = st.slider("模拟次数 N", min_value=100, max_value=20000, value=2000, step=100, key="ex_N")

    # 计算参考值（使用数值积分）
    try:
        true_value_ex, error_est = spi.quad(f_ex, a_ex, b_ex)
        has_true_value = True
    except:
        true_value_ex = None
        has_true_value = False
        st.warning("无法计算精确积分值作为参考，将只显示蒙特卡洛估计")

    # 运行蒙特卡洛积分
    x_samples_ex = np.random.uniform(a_ex, b_ex, N_ex)
    f_values_ex = f_ex(x_samples_ex)
    integral_estimate_ex = (b_ex - a_ex) * np.mean(f_values_ex)

    # 计算误差和置信区间
    f_std_ex = np.std(f_values_ex, ddof=1)
    se_ex = (b_ex - a_ex) * f_std_ex / np.sqrt(N_ex)
    ci_lower_ex = integral_estimate_ex - 1.96 * se_ex
    ci_upper_ex = integral_estimate_ex + 1.96 * se_ex

    # 显示结果
    st.subheader("计算结果")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("蒙特卡洛估计", f"{integral_estimate_ex:.8f}")

    with col2:
        if has_true_value:
            error_ex = abs(integral_estimate_ex - true_value_ex)
            st.metric("绝对误差", f"{error_ex:.8f}")

    with col3:
        if has_true_value:
            rel_error_ex = 100 * error_ex / abs(true_value_ex)
            st.metric("相对误差", f"{rel_error_ex:.4f}%")

    st.metric("95%置信区间", f"[{ci_lower_ex:.8f}, {ci_upper_ex:.8f}]")

    # 可视化
    st.subheader("可视化")

    # 函数曲线和随机点
    x_curve_ex = np.linspace(a_ex, b_ex, 1000)
    y_curve_ex = f_ex(x_curve_ex)

    fig_ex = go.Figure()

    # 函数曲线
    fig_ex.add_trace(go.Scatter(
        x=x_curve_ex,
        y=y_curve_ex,
        mode='lines',
        name=f'f(x)',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)'
    ))

    # 随机样本点
    fig_ex.add_trace(go.Scatter(
        x=x_samples_ex[:300],  # 只显示部分点
        y=f_values_ex[:300],
        mode='markers',
        name='随机样本',
        marker=dict(size=5, color='red', opacity=0.6)
    ))

    fig_ex.update_layout(
        title=f"蒙特卡洛积分: ∫_{{{a_ex}}}^{{{b_ex}}} f(x) dx",
        xaxis_title='x',
        yaxis_title='f(x)',
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig_ex, use_container_width=True)

    # 收敛性测试
    st.subheader("收敛性测试")

    # 测试不同N值下的表现
    N_test_ex = np.logspace(2, 4.3, 15).astype(int)
    estimates_ex = []
    errors_ex = []
    ci_widths_ex = []

    for n in N_test_ex:
        x_temp = np.random.uniform(a_ex, b_ex, n)
        f_temp = f_ex(x_temp)
        est = (b_ex - a_ex) * np.mean(f_temp)
        estimates_ex.append(est)

        if has_true_value:
            errors_ex.append(abs(est - true_value_ex))

        # 计算置信区间宽度
        std_temp = np.std(f_temp, ddof=1)
        ci_width = 2 * 1.96 * (b_ex - a_ex) * std_temp / np.sqrt(n)
        ci_widths_ex.append(ci_width)

    # 创建收敛图
    fig_convergence_ex = go.Figure()

    fig_convergence_ex.add_trace(go.Scatter(
        x=N_test_ex,
        y=estimates_ex,
        mode='lines+markers',
        name='蒙特卡洛估计',
        line=dict(color='blue')
    ))

    if has_true_value:
        fig_convergence_ex.add_hline(y=true_value_ex, line_dash="dash", line_color="red",
                                     name="参考值")

    fig_convergence_ex.add_trace(go.Scatter(
        x=N_test_ex,
        y=ci_widths_ex,
        mode='lines+markers',
        name='置信区间宽度',
        line=dict(color='green'),
        yaxis='y2'
    ))

    fig_convergence_ex.update_layout(
        title="蒙特卡洛积分收敛性",
        xaxis=dict(title="模拟次数 N", type="log"),
        yaxis=dict(title="积分估计值"),
        yaxis2=dict(title="置信区间宽度", overlaying="y", side="right"),
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_convergence_ex, use_container_width=True)

    # 练习问题
    st.subheader("思考问题")

    with st.expander("问题1：如何提高蒙特卡洛积分的精度？"):
        st.markdown("""
        **提高蒙特卡洛积分精度的方法**：

        1. **增加样本量N**：误差以 $O(1/\\sqrt{N})$ 收敛，增加N是最直接的方法

        2. **方差缩减技术**：
           - **重要抽样**：根据函数形状调整抽样分布
           - **对偶变量**：使用负相关的随机变量对
           - **控制变量**：用已知期望的变量减少方差
           - **分层抽样**：将积分区域划分为子区域

        3. **准蒙特卡洛方法**：使用低差异序列代替随机数

        4. **自适应蒙特卡洛**：根据函数变化调整抽样密度
        """)

    with st.expander("问题2：为什么蒙特卡洛方法在高维积分中更有优势？"):
        st.markdown("""
        **维度优势**：

        1. **收敛速度与维度无关**：蒙特卡洛误差为 $O(1/\\sqrt{N})$，与维度d无关
           - 传统数值方法（如梯形法则）误差为 $O(N^{-2/d})$
           - 当d较大时，传统方法需要极多的采样点

        2. **计算复杂度**：
           - 蒙特卡洛：计算量与维度成线性关系 $O(dN)$
           - 网格方法：计算量与 $N^d$ 成指数关系

        3. **实现简单性**：高维情况下，随机抽样比构造高维网格简单得多

        4. **适用性广**：可处理复杂形状的高维区域，传统方法难以处理
        """)

# ============================================
# 第六部分：Q&A环节
# ============================================
elif selected_case == "Q&A环节":
    st.markdown('<h2 class="section-header">常见问题解答 (Q&A)</h2>', unsafe_allow_html=True)

    # 问题列表
    questions = [
        {
            "question": "模拟次数N如何影响精度？误差与N的关系是什么？",
            "answer": """
            **精度与N的关系**：

            1. **误差收敛速度**：蒙特卡洛方法的误差以 $O(1/\\sqrt{N})$ 的速度收敛
               - 这意味着要使误差减半，需要将N增加到原来的4倍

            2. **中心极限定理的应用**：
               $$
               \\hat{I}_N \\sim \\mathcal{N}\\left(I, \\frac{\\sigma^2}{N}\\right)
               $$
               其中 $\\sigma^2 = \\text{Var}[f(X)]$ 是函数值的方差

            3. **95%置信区间宽度**：
               $$
               \\text{CI宽度} = 2 \\times 1.96 \\times \\frac{\\sigma}{\\sqrt{N}} \\propto \\frac{1}{\\sqrt{N}}
               $$

            4. **实践建议**：
               - 对于大多数应用，N=10,000~100,000可获得合理精度
               - 可通过预实验估计σ，然后根据所需精度确定N
            """
        },
        {
            "question": "为什么蒙特卡洛在高维问题中更有优势？",
            "answer": """
            **高维优势（维度诅咒的规避）**：

            1. **收敛速度独立于维度**：
               - 蒙特卡洛：$\\text{误差} \\propto 1/\\sqrt{N}$（与维度d无关）
               - 网格方法：$\\text{误差} \\propto N^{-k/d}$（随d增加急剧恶化）

            2. **计算复杂度对比**：
               - d维积分，要达到精度ε：
               - 蒙特卡洛：$N \\propto 1/\\epsilon^2$，计算量 $O(\\epsilon^{-2})$
               - 传统数值积分：$N \\propto \\epsilon^{-d/k}$，计算量 $O(\\epsilon^{-d/k})$
               - 当d>2k时，蒙特卡洛更高效

            3. **实现复杂性**：
               - 高维网格难以构造和存储
               - 蒙特卡洛只需生成随机点，实现简单

            4. **应用领域**：
               - 金融工程（期权定价，维度可达数百）
               - 统计物理（相空间积分）
               - 机器学习（高维积分和优化）
            """
        },
        {
            "question": "随机数质量对蒙特卡洛模拟重要吗？",
            "answer": """
            **随机数质量至关重要**：

            1. **伪随机数生成器(PRNG)的要求**：
               - 长周期：避免序列重复
               - 均匀性：在[0,1]^d空间中均匀分布
               - 独立性：序列中无明显相关性
               - 可重复性：便于调试和验证

            2. **常见问题**：
               - **相关性**：导致估计偏差
               - **短周期**：大样本时出现模式重复
               - **低差异**：影响均匀性

            3. **推荐方案**：
               - **Mersenne Twister**：周期长($2^{19937}-1$)，统计性质好
               - **Sobol序列**：准随机数，收敛更快
               - **加密安全PRNG**：用于高安全性需求

            4. **质量测试**：
               - 统计检验：卡方检验、KS检验等
               - 经验检验：可视化检查、多维均匀性测试
            """
        },
        {
            "question": "如何为建模问题选择合适的概率分布？",
            "answer": """
            **分布选择策略**：

            1. **基于问题性质**：
               - **计数过程**：泊松分布（到达次数）
               - **等待时间**：指数分布（无记忆性）
               - **连续测量**：正态分布（中心极限定理）
               - **有界变量**：Beta分布（比例问题）

            2. **基于先验知识**：
               - **历史数据**：经验分布
               - **理论推导**：从基本原理推导
               - **专家判断**：主观概率分布

            3. **分布拟合步骤**：
               1. 数据收集或理论分析
               2. 可视化探索（直方图、Q-Q图）
               3. 候选分布选择
               4. 参数估计（极大似然、矩估计）
               5. 拟合优度检验（KS检验、卡方检验）

            4. **稳健性考虑**：
               - 使用重尾分布应对异常值
               - 敏感性分析：测试不同分布的影响
            """
        },
        {
            "question": "蒙特卡洛方法的收敛速度如何？有哪些加速技巧？",
            "answer": """
            **收敛速度与加速技术**：

            1. **基本收敛速度**：
               - 普通蒙特卡洛：$O(1/\\sqrt{N})$
               - 准蒙特卡洛：$O((\\log N)^d / N)$（低差异序列）

            2. **方差缩减技术**：
               - **重要抽样**：
                 $$\\text{Var}_{IS} = \\frac{1}{N}\\text{Var}\\left(\\frac{f(X)}{g(X)}\\right)$$
                 选择g(x)使f(x)/g(x)近似常数

               - **对偶变量**：利用负相关性
                 $$\\hat{I}_{AV} = \\frac{1}{2N}\\sum_{i=1}^N [f(U_i) + f(1-U_i)]$$

               - **控制变量法**：
                 $$\\hat{I}_{CV} = \\frac{1}{N}\\sum f(X_i) - \\beta\\left(\\frac{1}{N}\\sum h(X_i) - \\mathbb{E}[h(X)]\\right)$$

               - **分层抽样**：分层内方差较小

            3. **其他加速方法**：
               - **并行计算**：蒙特卡洛天然可并行化
               - **GPU加速**：适合大规模简单模拟
               - **多级蒙特卡洛**：不同精度组合
            """
        },
        {
            "question": "蒙特卡洛方法的主要局限性是什么？",
            "answer": """
            **局限性与应对策略**：

            1. **收敛速度慢**：
               - 问题：需要大量样本获得高精度
               - 应对：使用方差缩减技术，准蒙特卡洛

            2. **随机性本身的问题**：
               - 问题：结果有随机波动
               - 应对：计算置信区间，多次运行取平均

            3. **高方差问题**：
               - 问题：函数变化剧烈时方差大
               - 应对：重要抽样，自适应抽样

            4. **稀有事件模拟**：
               - 问题：小概率事件需要极多样本
               - 应对：重要性抽样，分裂法，子集模拟

            5. **适用性限制**：
               - 问题：需要可计算的概率模型
               - 应对：与其他方法结合（如MCMC）

            6. **计算成本**：
               - 问题：复杂模型每次模拟成本高
               - 应对：代理模型，多保真度方法
            """
        }
    ]

    # 显示所有问题和答案
    for i, qa in enumerate(questions):
        with st.expander(f"Q{i + 1}: {qa['question']}"):
            st.markdown(qa['answer'])

    # 总结
    st.markdown("---")
    st.markdown("""
    ### 总结

    蒙特卡洛方法是一种强大的数值计算工具，它将确定性问题转化为随机模拟问题，通过大量随机抽样获得近似解。

    **关键要点**：
    1. 蒙特卡洛方法基于大数定律和中心极限定理
    2. 误差以 $1/\\sqrt{N}$ 收敛，与问题维度无关
    3. 适用于高维、复杂边界、无解析解的问题
    4. 方差缩减技术可显著提高效率

    **学习建议**：
    - 从简单问题开始（如π估计），理解基本思想
    - 实践不同应用场景（积分、优化、模拟）
    - 学习方差缩减技术提高效率
    - 理解方法的局限性，知道何时使用

    继续探索蒙特卡洛方法的世界，你会发现它在金融、物理、工程、人工智能等众多领域的广泛应用！
    """)

# ============================================
# 页脚
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>蒙特卡洛模拟教学网页 | 设计用于数学建模教育 | © 2024</p>
    <p>适用对象：具有基础统计学和概率论知识的学生</p>
    <p>推荐学习路径：理论 → 案例1 → 案例2 → 案例3 → 练习 → Q&A</p>
</div>
""", unsafe_allow_html=True)
