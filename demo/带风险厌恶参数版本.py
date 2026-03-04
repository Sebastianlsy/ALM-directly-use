"""
================================================================================
资产配置优化脚本 —— 详细注释版
================================================================================

【脚本功能总览】
本脚本实现了基于 Markowitz 均值-方差模型的资产组合优化。
核心目标：在给定一组资产的预期收益率和协方差矩阵后，找到"有效前沿"（Efficient Frontier），
即在每个风险水平下收益最高（或在每个收益水平下风险最低）的最优组合集合。

具体实现了两种求解有效前沿的方法，并进行对比：
  方法一：参数化追踪法 —— 固定目标收益率，逐点求最小波动率
  方法二：风险厌恶系数扫描法 —— 扫描不同的风险厌恶参数 λ，求最大化均值-方差效用

最终输出：
  1. 三张并排的可视化图（参数化法 / λ扫描法 / 两者叠加对比）
  2. 最大夏普比率组合（切线组合）的权重配置
  3. 两种方法的数值对比

【依赖库】
  - pandas:     数据读取与表格操作
  - numpy:      矩阵运算与数值计算
  - matplotlib: 绑图可视化
  - scipy:      数值优化（SLSQP 求解器）

【输入文件】
  - 中间项.xlsx:     包含各资产的市值、预期收益率等量化指标
  - 相关性矩阵.xlsx: 包含资产间的相关性矩阵和各资产的波动率

【核心数学原理】
  Markowitz 均值-方差框架：
    - 组合收益率:  R_p = w^T × μ          （权重向量 × 预期收益率向量）
    - 组合方差:    σ²_p = w^T × Σ × w      （权重向量 × 协方差矩阵 × 权重向量）
    - 夏普比率:    SR = (R_p - R_f) / σ_p   （超额收益 / 波动率）
    - 有效前沿:    在所有可行组合中，给定风险下收益最高的组合轨迹
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


# ==================== 0. 中文字体设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

# 解决 matplotlib 使用中文字体时负号 "-" 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 全局参数配置 ====================

# 输入文件路径
file_main      = "中间项.xlsx"         # 主数据文件，包含资产量化指标
file_corr      = "相关性矩阵.xlsx"     # 相关性矩阵和波动率数据

# 目标账户：从主数据文件中筛选该账户下的资产
target_account = '万能账户'

# 无风险利率（Risk-Free Rate）：用于计算夏普比率
# 夏普比率 = (组合收益率 - 无风险利率) / 组合波动率
# 这里设为 2%，通常可用国债收益率近似
rf_rate = 0.02

# 随机组合数量
num_portfolios = 100000

# 资产名称映射表：统一不同数据源中同一资产的名称
# 例如 "高等级信用债_3Y" 在不同表中可能叫 "信用债_3Y"
# 通过映射确保三张表（量化指标、相关性矩阵、波动率）能正确对齐
name_map = {
    "高等级信用债_3Y":  "信用债_3Y",
    "高等级信用债_5Y":  "信用债_5Y",
    "高等级信用债_10Y": "信用债_10Y",
}


# ==================== 2. 数据读取辅助函数 ====================

def read_asset_rows(target_account):
    """
    从主数据文件中读取指定账户的资产行。

    【处理流程】
    1. 读取 '量化指标' 工作表
    2. 筛选条件（四重过滤）：
       a) 保险账户分类 == target_account（只要目标账户）
       b) 第一列不含 "负债"（排除负债端数据，只保留资产端）
       c) 全价市场价值 > 0（排除已清仓或无效记录）
       d) 三级资产分类非空（排除未分类的数据）
    3. 对资产名称做标准化映射（通过 name_map）

    【参数】
    target_account: str — 目标保险账户名称，如 '万能账户'

    【返回】
    df_assets: DataFrame — 筛选并标准化后的资产数据
               新增列 'asset_std' 为标准化后的资产名称
    """
    # 读取 Excel 文件中的 "量化指标" 工作表
    df_quant = pd.read_excel(file_main, sheet_name='量化指标')

    # 构建筛选条件掩码（布尔索引）
    mask = (
        # 条件 a：只选目标账户
        (df_quant['保险账户分类/InsuranceAccountType'] == target_account) &
        # 条件 b：排除负债行（第一列包含"负债"字样的行）
        (~df_quant.iloc[:, 0].str.contains('负债', na=False)) &
        # 条件 c：市场价值为正（有效持仓）
        (df_quant['全价市场价值/DirtyMarketValue'] > 0) &
        # 条件 d：三级分类不为空（确保能归入某个资产类别）
        (df_quant['资产配置分类三级/SAAAssetTypeLevel3'].notna())
    )

    # 应用掩码筛选，.copy() 避免 SettingWithCopyWarning
    df_assets = df_quant[mask].copy()

    # 检查是否有数据
    if df_assets.empty:
        raise ValueError(f"未找到账户 [{target_account}] 的资产")

    # 对三级资产分类名称做标准化映射
    # name_map.get(x, x) 的逻辑：如果 x 在映射表中则返回映射值，否则返回原值
    df_assets['asset_std'] = df_assets['资产配置分类三级/SAAAssetTypeLevel3'].map(
        lambda x: name_map.get(x, x)
    )

    return df_assets


def calc_weighted_avg_return(df_assets):
    """
    按资产类别计算加权平均预期收益率。

    【为什么要加权？】
    同一资产类别（如 "信用债_3Y"）下可能有多只券，
    每只券的市场价值和预期收益率不同。
    以市场价值为权重计算加权平均，使得大持仓的券对组合收益率的影响更大，
    这比简单算术平均更准确地反映该类别的实际收益贡献。

    【参数】
    df_assets: DataFrame — read_asset_rows() 返回的资产数据

    【返回】
    DataFrame — 索引为资产名称，列 'mu' 为加权平均预期收益率
    """
    result = {}
    # 按标准化后的资产名称分组
    for asset_name, group in df_assets.groupby('asset_std'):
        # 提取该类别下所有券的市场价值（作为权重）
        market_values   = group['全价市场价值/DirtyMarketValue'].values
        # 提取预期收益率，缺失值填 0
        returns         = group['预期投资收益率/ExpectedReturn'].fillna(0).values
        # 以市场价值为权重计算加权平均收益率
        # weighted_return = Σ(w_i × r_i)，其中 w_i = MV_i / Σ(MV)
        weighted_return = np.average(returns, weights=market_values)
        result[asset_name] = weighted_return

    # 转为 DataFrame，orient='index' 表示字典的键作为行索引
    return pd.DataFrame.from_dict(result, orient='index', columns=['mu'])


def read_corr_matrix():
    """
    读取并清洗资产间的相关性矩阵。

    【相关性矩阵要求】
    - 必须是方阵（N×N），行列索引为资产名称
    - 对角线为 1（资产与自身完全正相关）
    - 对称：corr(A,B) = corr(B,A)
    - 所有元素在 [-1, 1] 范围内

    【处理流程】
    1. 读取原始矩阵
    2. 剔除负债相关行列
    3. 处理 NaN：利用对称性填充（如果 corr[i,j] 是 NaN 但 corr[j,i] 有值，
       则用 corr[j,i] 填充）
    4. 剩余 NaN 填 0（假设无相关性）
    5. 对角线强制设为 1

    【返回】
    DataFrame — 清洗后的相关性矩阵
    """
    # 读取相关性矩阵，第一列作为行索引
    df_corr_raw = pd.read_excel(file_corr, sheet_name='相关性', index_col=0)

    # 剔除负债相关的行和列（只关注资产端）
    drop_names  = [name for name in df_corr_raw.index if '负债' in str(name)]
    df_corr_raw = df_corr_raw.drop(index=drop_names, columns=drop_names, errors='ignore')

    # 转为 numpy 数组进行数值处理
    arr = df_corr_raw.to_numpy(dtype=float, copy=True)

    # 利用对称性填充 NaN：
    # 如果 arr[i,j] 是 NaN 但 arr[j,i] 有值，则用 arr[j,i] 填充
    # 处理了数据录入时只填了上三角或下三角的情况
    nan_positions = np.isnan(arr)
    arr[nan_positions] = arr.T[nan_positions]

    # 仍然为 NaN 的位置（即 arr[i,j] 和 arr[j,i] 都缺失），填 0
    # 假设：缺失相关性的资产对之间不相关
    arr = np.nan_to_num(arr, nan=0.0)

    # 对角线强制为 1（资产与自身的相关系数恒为 1）
    np.fill_diagonal(arr, 1.0)

    return pd.DataFrame(arr, index=df_corr_raw.index, columns=df_corr_raw.columns)


def read_vol_series():
    """
    读取各资产的年化波动率（标准差）。

    【波动率的含义】
    波动率 σ 衡量资产收益率的离散程度（不确定性/风险）。
    年化波动率 = 日波动率 × √252（交易日数）
    例如：σ = 0.05 表示该资产年化波动率为 5%

    【返回】
    Series — 索引为资产名称，值为年化波动率
    """
    # 读取 '自定义指标' 工作表
    df_vol = pd.read_excel(file_corr, sheet_name='自定义指标')

    # 将第一列设为索引，取第二列的值（即波动率数据）
    vol_series = df_vol.set_index(df_vol.columns[0]).iloc[:, 0]

    # 剔除负债相关行
    drop_names = [name for name in vol_series.index if '负债' in str(name)]
    return vol_series.drop(index=drop_names, errors='ignore')


def build_cov_matrix(common, corr_matrix, vol_series):
    """
    由相关性矩阵和波动率构建协方差矩阵。

    【数学原理】
    协方差矩阵 Σ 的构建公式：
        Σ = diag(σ) × C × diag(σ)

    其中：
        - diag(σ) 是对角阵，对角线元素为各资产的波动率
        - C 是相关性矩阵
        - σ_i 是资产 i 的波动率

    展开后：Σ[i,j] = σ_i × C[i,j] × σ_j
    即：两资产的协方差 = 资产i波动率 × 相关系数 × 资产j波动率

    【正定性检查】
    协方差矩阵理论上必须是半正定的（所有特征值 ≥ 0），
    否则 w^T × Σ × w 可能为负（方差不可能为负，物理意义不成立）。
    如果因数值误差导致出现负特征值，需要修正。

    【参数】
    common:      list — 参与优化的资产名称列表
    corr_matrix: DataFrame — 相关性矩阵
    vol_series:  Series — 波动率序列

    【返回】
    cov: ndarray — 协方差矩阵（N×N）
    """
    # 提取参与优化的资产的波动率向量
    sigma = vol_series.loc[common].values

    # 提取对应的相关性子矩阵
    corr  = corr_matrix.loc[common, common].values

    # 构建协方差矩阵：Σ = diag(σ) @ C @ diag(σ)
    # np.diag(sigma) 创建对角矩阵，@ 是矩阵乘法运算符
    cov = np.diag(sigma) @ corr @ np.diag(sigma)

    # 检查正定性：计算所有特征值
    eigenvalues = np.linalg.eigvals(cov)
    if np.any(eigenvalues < -1e-10):
        # 如果存在显著的负特征值，通过加小量单位阵修正
        # 修正量 = |最小负特征值| + 微小正数，确保修正后所有特征值 > 0
        print("警告：协方差矩阵非正定，已修正")
        cov += (-eigenvalues.min() + 1e-8) * np.eye(len(common))

    return cov


def load_data(target_account):
    """
    数据加载主函数：读取三张表并对齐，返回优化所需的向量和矩阵。

    【三表对齐逻辑】
    三个数据源各自包含一组资产名称：
      - 量化指标表（预期收益率）
      - 相关性矩阵（资产间相关系数）
      - 波动率表（各资产标准差）
    取三者的交集（intersection），只对同时出现在三张表中的资产进行优化。
    这样避免了因某资产缺少波动率或相关性数据而导致优化失败。

    【参数】
    target_account: str — 目标保险账户名称

    【返回】
    common: list    — 参与优化的资产名称列表（长度 N）
    mu:     ndarray — 预期收益率向量（长度 N）
    cov:    ndarray — 协方差矩阵（N×N）
    """
    # 第一步：读取并汇总资产数据
    df_assets  = read_asset_rows(target_account)
    df_summary = calc_weighted_avg_return(df_assets)

    # 第二步：读取相关性矩阵和波动率
    corr_matrix = read_corr_matrix()
    vol_series  = read_vol_series()

    # 第三步：取三张表资产名称的交集
    common = list(
        df_summary.index                   # 资产收益率表中的资产
        .intersection(corr_matrix.index)   # ∩ 相关性矩阵中的资产
        .intersection(vol_series.index)    # ∩ 波动率表中的资产
    )
    if not common:
        raise ValueError("资产名称三表无法对齐")

    print(f"[资产] 参与优化 {len(common)} 项：{', '.join(common)}")

    # 提取对齐后的数据
    mu    = df_summary.loc[common, 'mu'].values   # 预期收益率向量
    cov   = build_cov_matrix(common, corr_matrix, vol_series)  # 协方差矩阵
    sigma = vol_series.loc[common].values          # 波动率向量（用于打印）

    # 打印资产摘要表
    print(f"\n{'资产':<18} {'预期收益率':>10} {'波动率':>10}")
    print("-" * 42)
    for a, m, s in zip(common, mu, sigma):
        print(f"{a:<18} {m:>10.4f} {s:>10.4f}")

    return common, mu, cov


# ==================== 3. 狄利克雷随机采样（蒙特卡洛模拟） ====================

def dirichlet_sampling(mu, cov):
    """
    使用狄利克雷分布在权重单纯形上均匀随机采样，生成大量随机组合。
    主要用于可视化：在风险-收益平面上形成散点云，展示所有可行组合的分布范围。

    【什么是权重单纯形？】
    N 个资产的权重向量 w = (w_1, w_2, ..., w_N) 满足：
      - w_i ≥ 0（不允许做空）
      - Σw_i = 1（权重之和为 100%）
    满足这两个条件的所有 w 构成一个 (N-1) 维单纯形。

    【什么是狄利克雷分布？】
    Dirichlet(α_1, α_2, ..., α_N) 是定义在单纯形上的概率分布。
    当所有 α_i = 1 时，退化为单纯形上的均匀分布，
    即每种权重组合被采样到的概率相同。
    这保证散点云均匀覆盖整个可行域，不会偏向某些特定的资产配比。

    【参数】
    mu:  ndarray — 预期收益率向量（长度 N）
    cov: ndarray — 协方差矩阵（N×N）

    【返回】
    port_ret:    ndarray — 各随机组合的预期收益率（长度 num_portfolios）
    port_vol:    ndarray — 各随机组合的波动率（长度 num_portfolios）
    port_sharpe: ndarray — 各随机组合的夏普比率（长度 num_portfolios）
    """
    n = len(mu)

    # 生成 num_portfolios 个随机权重向量
    # [1.0] * n 表示所有 α 参数都为 1，即单纯形上的均匀分布
    # 返回矩阵 W 的形状为 (num_portfolios, n)，每行是一个权重向量
    W = np.random.dirichlet([1.0] * n, num_portfolios)

    # 计算每个组合的预期收益率：R_p = w^T × μ
    # W @ mu 等价于对 W 的每一行与 mu 做点积
    port_ret = W @ mu

    # 计算每个组合的波动率：σ_p = sqrt(w^T × Σ × w)
    # np.einsum('ij,jk,ik->i', W, cov, W) 是高效的批量二次型计算：
    #   对于第 i 个组合：result[i] = Σ_j Σ_k W[i,j] × cov[j,k] × W[i,k]
    # 这等价于逐行计算 W[i] @ cov @ W[i]，但向量化后速度快很多
    port_vol = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))

    # 计算夏普比率：SR = (R_p - R_f) / σ_p
    port_sharpe = (port_ret - rf_rate) / port_vol

    return port_ret, port_vol, port_sharpe


# ==================== 4. 方法一：参数化追踪法求有效前沿 ====================

def calc_frontier_parametric(mu, cov, n_points=60):
    """
    【参数化追踪法（Target Return Sweep）】

    核心思路：
    在预期收益率的范围内均匀取 n_points 个目标值，
    对每个目标收益率，求解以下优化问题：

        min  σ_p = sqrt(w^T × Σ × w)       ← 最小化组合波动率
        s.t. Σw_i = 1                        ← 权重之和为 1
             w^T × μ = target_return         ← 组合收益率等于目标值
             w_i ≥ 0                          ← 不允许做空

    将这些最优点连成曲线，就是有效前沿。

    【为什么只保留上半段？】
    完整的最小方差曲线是一个"子弹形"（或抛物线形），
    下半段（最小方差点以下）的组合是"低效"的——
    存在同样风险但收益更高的组合。
    有效前沿只包含上半段。

    【多起点策略】
    SLSQP 是局部优化器，可能陷入局部最优。
    对每个目标收益率尝试 5 个随机初始点，取最优结果。

    【参数】
    mu:       ndarray — 预期收益率向量
    cov:      ndarray — 协方差矩阵
    n_points: int     — 前沿上的采样点数（默认 60）

    【返回】
    frontier_vols: ndarray — 有效前沿上各点的波动率
    frontier_rets: ndarray — 有效前沿上各点的预期收益率
    """
    n = len(mu)
    bounds = [(0, 1)] * n   # 每个资产权重在 [0, 1] 之间（不做空，不超过100%）

    # 生成目标收益率序列
    # 从最低资产收益率到最高资产收益率的 98%（留余量，避免边界处无解）
    target_rets = np.linspace(mu.min(), mu.max() * 0.98, n_points)

    frontier_vols = []   # 存储每个目标收益率对应的最小波动率
    frontier_rets = []   # 存储成功求解的目标收益率

    for target_ret in target_rets:
        # 定义约束条件
        constraints = [
            # 约束 1：权重之和 = 1
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            # 约束 2：组合收益率 = 目标收益率
            # 注意 tr=target_ret 是闭包绑定，避免 Python lambda 的延迟绑定问题
            {'type': 'eq', 'fun': lambda w, tr=target_ret: w @ mu - tr},
        ]

        # 多起点策略：5 次随机初始化，取最优解
        best = None
        for _ in range(5):
            # 用狄利克雷分布生成满足权重约束的随机初始点
            w0 = np.random.dirichlet([1.0] * n)

            # 使用 SLSQP（序列二次规划法）求解
            # 目标函数：最小化组合波动率 sqrt(w^T × Σ × w)
            res = minimize(
                lambda w: np.sqrt(w @ cov @ w),    # 目标函数
                w0,                                  # 初始权重
                method='SLSQP',                      # 支持等式/不等式约束的求解器
                bounds=bounds,                       # 变量边界
                constraints=constraints,             # 等式约束
                options={
                    'ftol': 1e-12,                   # 函数值收敛精度（非常高精度）
                    'maxiter': 1000                  # 最大迭代次数
                }
            )
            # 保留最优解：求解成功且目标函数值更小
            if res.success and (best is None or res.fun < best.fun):
                best = res

        # 如果该目标收益率有成功的解，记录结果
        if best is not None and best.success:
            frontier_vols.append(best.fun)       # 最小波动率
            frontier_rets.append(target_ret)     # 对应的目标收益率

    frontier_vols = np.array(frontier_vols)
    frontier_rets = np.array(frontier_rets)

    # 只保留最小方差点以上的上半段（有效前沿）
    if len(frontier_vols) > 1:
        mi            = np.argmin(frontier_vols)   # 找到全局最小方差点
        frontier_vols = frontier_vols[mi:]         # 从最小方差点往右取
        frontier_rets = frontier_rets[mi:]

    return frontier_vols, frontier_rets


# ==================== 5. 方法二：风险厌恶系数扫描法求有效前沿 ====================

def calc_frontier_lambda(mu, cov, assets):
    """
    【风险厌恶系数扫描法（Risk Aversion Sweep）】

    核心思路：
    引入风险厌恶系数 λ（lambda），对不同的 λ 值求解：

        max  U(w) = w^T × μ - (λ/2) × w^T × Σ × w
        s.t. Σw_i = 1
             w_i ≥ 0

    其中 U(w) 是"均值-方差效用函数"：
      - 第一项 w^T × μ 是组合收益率（越大越好）
      - 第二项 (λ/2) × w^T × Σ × w 是风险惩罚（方差的 λ/2 倍）
      - λ 越大，投资者越厌恶风险，最优组合越偏向低波动
      - λ 越小，投资者越追求收益，最优组合越偏向高收益

    【λ 与有效前沿的关系】
    每个 λ 值对应有效前沿上唯一的一个点：
      λ → 0:   最优点趋近于前沿右上端（纯高收益资产）
      λ → ∞:   最优点趋近于全局最小方差组合（前沿左下端）
    扫描 λ 从小到大，就能逐点描出整条有效前沿。

    【为什么用对数间隔扫描？】
    λ 与前沿位置不是线性关系。低 λ 区域（激进端）变化更敏感，
    需要更密的采样。np.logspace(-1, 2.3, 60) 生成从 0.1 到 ~200 的
    60 个对数等间距值，低端密、高端疏，与前沿曲率匹配。

    【局部散点云】
    以每个 λ 的最优权重为中心，用集中型狄利克雷分布在其附近撒点。
    与全局均匀撒点不同，这种策略让散点集中在有效前沿附近，
    更好地展示前沿附近的组合分布。

    集中型狄利克雷的参数 α = w_opt × concentration + 0.1：
      - w_opt 大的资产 → α 大 → 撒点时该资产权重倾向于大
      - concentration 控制集中程度：越大散点越贴近最优点
      - +0.1 防止权重为 0 的资产 α=0 导致数值错误

    【参数】
    mu:     ndarray — 预期收益率向量
    cov:    ndarray — 协方差矩阵
    assets: list    — 资产名称列表

    【返回】
    all_vols:       ndarray — 所有局部散点的波动率
    all_rets:       ndarray — 所有局部散点的收益率
    all_sharpes:    ndarray — 所有局部散点的夏普比率
    optimal_points: list    — 每个 λ 的最优点信息字典列表
    """
    n = len(mu)

    # 对数间隔扫描 λ 值
    # 10^(-1) = 0.1（极激进）到 10^(2.3) ≈ 200（极保守）
    lambda_values = np.logspace(-1, 2.3, 60)

    optimal_points = []   # 存储每个 λ 的最优点
    all_vols       = []   # 所有局部散点的波动率
    all_rets       = []   # 所有局部散点的收益率
    all_sharpes    = []   # 所有局部散点的夏普比率

    print("扫描风险厌恶系数，计算有效前沿...")

    for lam in lambda_values:

        # ---- 求当前 λ 下的最优权重 ----

        def objective(w, lam=lam):
            """
            均值-方差效用函数（取负号变为最小化问题）。

            原问题：max U(w) = w^T×μ - (λ/2) × w^T×Σ×w
            等价于：min -U(w) = -(w^T×μ) + (λ/2) × w^T×Σ×w

            注意这里用方差（不开根号），而非标准差/波动率，
            这是均值-方差效用的标准定义。
            """
            ret      = w @ mu            # 组合收益率
            variance = w @ cov @ w       # 组合方差（σ²）
            return -(ret - (lam / 2) * variance)

        # 约束和边界
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds      = [(0, 1)] * n

        # 20 次多起点策略（比参数化法更多次，因为效用函数景观可能更复杂）
        best = None
        for _ in range(20):
            w0  = np.random.dirichlet([1.0] * n)
            res = minimize(
                objective, w0, method='SLSQP', bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )
            if res.success and (best is None or res.fun < best.fun):
                best = res

        if best is None:
            continue   # 该 λ 值求解失败，跳过

        # 提取最优解的指标
        w_opt  = best.x                          # 最优权重向量
        r_opt  = w_opt @ mu                      # 组合收益率
        v_opt  = np.sqrt(w_opt @ cov @ w_opt)    # 组合波动率
        sr_opt = (r_opt - rf_rate) / v_opt        # 夏普比率

        optimal_points.append({
            'lambda':  lam,
            'weights': w_opt,
            'ret':     r_opt,
            'vol':     v_opt,
            'sharpe':  sr_opt
        })

        # ---- 以最优权重为中心，局部撒点 ----
        concentration = 20      # 集中度参数，值越大散点越贴近最优点
        alpha         = w_opt * concentration + 0.1   # 狄利克雷参数向量
        n_local       = 500     # 每个 λ 生成 500 个局部随机组合

        # 集中型狄利克雷采样：散点云围绕最优点分布
        W_local       = np.random.dirichlet(alpha, n_local)
        local_rets    = W_local @ mu
        local_vols    = np.sqrt(np.einsum('ij,jk,ik->i', W_local, cov, W_local))
        local_sharpes = (local_rets - rf_rate) / local_vols

        # 将局部散点加入全局列表
        all_rets.extend(local_rets)
        all_vols.extend(local_vols)
        all_sharpes.extend(local_sharpes)

    return (
        np.array(all_vols),
        np.array(all_rets),
        np.array(all_sharpes),
        optimal_points
    )


# ==================== 6. 求最大夏普比率组合（切线组合） ====================

def find_tangency_portfolio(mu, cov):
    """
    求最大夏普比率组合，又称"切线组合"（Tangency Portfolio）。

    【什么是切线组合？】
    在风险-收益平面上，从无风险利率点 (0, R_f) 向有效前沿画一条切线，
    切点对应的就是夏普比率最高的组合。
    这个组合在所有可行组合中，每承担一单位风险获得的超额收益最大。

    【优化问题】
        max  SR = (w^T×μ - R_f) / sqrt(w^T×Σ×w)
        s.t. Σw_i = 1
             w_i ≥ 0

    由于 SR 是分式（非凸），SLSQP 可能陷入局部最优，
    因此使用 20 次多起点策略提高全局最优的概率。

    【参数】
    mu:  ndarray — 预期收益率向量
    cov: ndarray — 协方差矩阵

    【返回】
    w_opt: ndarray — 最优权重向量
    r_opt: float   — 组合预期收益率
    v_opt: float   — 组合波动率
    """
    n = len(mu)

    def neg_sharpe(w):
        """目标函数：负夏普比率（最小化负值 = 最大化正值）"""
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf_rate) / vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n

    best = None
    for _ in range(20):
        w0  = np.random.dirichlet([1.0] * n)
        res = minimize(
            neg_sharpe, w0, method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        raise RuntimeError("最优点求解失败")

    w_opt = best.x
    r_opt = w_opt @ mu
    v_opt = np.sqrt(w_opt @ cov @ w_opt)
    return w_opt, r_opt, v_opt


# ==================== 7. 可视化：三图对比 ====================

def plot_both_frontiers(assets, mu, cov):
    """
    生成三张并排图，直观对比两种有效前沿求解方法。

    【三图布局】
    左图：参数化追踪法
      - 背景散点云：全局均匀采样的随机组合（颜色深浅 = 夏普比率高低）
      - 金色曲线：参数化法计算的有效前沿
      - 红色星号：最大夏普比率切线组合

    中图：λ 扫描法
      - 背景散点云：以各 λ 最优点为中心的局部采样（集中在前沿附近）
      - 金色曲线：λ 扫描法计算的有效前沿
      - 红色星号：λ 扫描法中夏普比率最高的点
      - 灰色标注：关键 λ 值的位置

    右图：两种方法叠加对比
      - 金色实线 vs 橙红虚线：两条前沿曲线
      - 两个星号：两种方法各自的最优点
      - 直接观察两条曲线是否重合

    【颜色编码】
    使用 'Blues' 色彩映射：
      - 浅蓝 = 低夏普比率（低效组合）
      - 深蓝 = 高夏普比率（高效组合）
    有效前沿就是深蓝散点的上边缘。

    【参数】
    assets: list    — 资产名称列表
    mu:     ndarray — 预期收益率向量
    cov:    ndarray — 协方差矩阵
    """

    # ---- 准备各类数据 ----
    print("生成全局散点云...")
    rets_global, vols_global, sharpes_global = dirichlet_sampling(mu, cov)

    print("计算参数化有效前沿...")
    fv_param, fr_param = calc_frontier_parametric(mu, cov)

    print("求切线组合...")
    w_opt, r_opt, v_opt = find_tangency_portfolio(mu, cov)
    sr_opt = (r_opt - rf_rate) / v_opt    # 切线组合夏普比率

    print("扫描风险厌恶系数...")
    vols_local, rets_local, sharpes_local, optimal_points = calc_frontier_lambda(mu, cov, assets)

    # 提取 λ 扫描法的有效前沿曲线数据
    fv_lambda = np.array([p['vol'] for p in optimal_points])
    fr_lambda = np.array([p['ret'] for p in optimal_points])

    # λ 扫描法中夏普比率最高的点
    best_point = max(optimal_points, key=lambda p: p['sharpe'])

    # ---- 创建三子图布局 ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f'有效前沿方法对比  —  {target_account}',
                 fontsize=15, fontweight='bold')

    # ==================== 左图：参数化追踪法 ====================
    ax1.set_title('参数化追踪法\n固定收益率 → 最小波动率', fontsize=11)

    # 散点云：全局随机组合，颜色映射夏普比率
    sc1 = ax1.scatter(vols_global, rets_global, c=sharpes_global,
                      cmap='Blues',      # 蓝色系色彩映射
                      alpha=0.3,         # 透明度 30%，避免遮挡前沿线
                      s=5)               # 点大小
    plt.colorbar(sc1, ax=ax1, label='夏普比率')   # 色标条

    # 有效前沿曲线
    if len(fv_param) > 1:
        ax1.plot(fv_param, fr_param, color='gold', linewidth=2.5,
                 zorder=8,               # 绘制层级：确保曲线在散点上方
                 label='有效前沿')

    # 最大夏普比率点（红色星号）
    ax1.scatter(v_opt, r_opt, color='red', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.5,
                label=f'最优点 SR={sr_opt:.3f}')

    ax1.set_xlabel('组合波动率', fontsize=11)
    ax1.set_ylabel('预期收益率', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)   # 虚线网格

    # ==================== 中图：λ 扫描法 ====================
    ax2.set_title('风险厌恶扫描法\n扫描 λ → 最大化均值方差效用', fontsize=11)

    # 散点云：局部采样（集中在前沿附近）
    sc2 = ax2.scatter(vols_local, rets_local, c=sharpes_local,
                      cmap='Blues', alpha=0.4, s=5)
    plt.colorbar(sc2, ax=ax2, label='夏普比率')

    # 有效前沿曲线
    if len(fv_lambda) > 1:
        ax2.plot(fv_lambda, fr_lambda, color='gold', linewidth=2.5,
                 zorder=8, label='有效前沿')

    # 标注几个关键 λ 值的位置，帮助理解 λ 与风险-收益的对应关系
    label_lambdas = [0.1, 1, 5, 20, 100]
    for p in optimal_points:
        # 只标注与关键值接近的点（相对误差 < 15%）
        if any(abs(p['lambda'] - lam) / lam < 0.15 for lam in label_lambdas):
            ax2.annotate(
                f"λ={p['lambda']:.1f}",
                xy=(p['vol'], p['ret']),                    # 箭头指向的位置
                xytext=(p['vol'] + 0.003, p['ret']),        # 文字位置（略偏右）
                fontsize=8, color='dimgray',
                arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.8)
            )

    # λ 扫描法的最优点
    ax2.scatter(best_point['vol'], best_point['ret'],
                color='red', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.5,
                label=f"最优点 SR={best_point['sharpe']:.3f}")

    ax2.set_xlabel('组合波动率', fontsize=11)
    ax2.set_ylabel('预期收益率', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ==================== 右图：两条前沿叠加对比 ====================
    ax3.set_title('两种方法有效前沿对比\n（曲线越接近说明两种方法越一致）', fontsize=11)

    # 参数化前沿：金色实线
    if len(fv_param) > 1:
        ax3.plot(fv_param, fr_param, color='gold', linewidth=2.5,
                 zorder=8, label='参数化追踪法', linestyle='-')

    # λ 扫描前沿：橙红色虚线
    if len(fv_lambda) > 1:
        ax3.plot(fv_lambda, fr_lambda, color='tomato', linewidth=2.5,
                 zorder=9, label='λ 扫描法', linestyle='--')

    # 两种方法各自的最优点
    ax3.scatter(v_opt, r_opt, color='gold', marker='*', s=300,
                zorder=10, edgecolors='black', linewidths=0.5,
                label=f'参数化最优 SR={sr_opt:.3f}')

    ax3.scatter(best_point['vol'], best_point['ret'],
                color='tomato', marker='*', s=300,
                zorder=10, edgecolors='black', linewidths=0.5,
                label=f"λ扫描最优 SR={best_point['sharpe']:.3f}")

    ax3.set_xlabel('组合波动率', fontsize=11)
    ax3.set_ylabel('预期收益率', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('frontier_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 frontier_comparison.png")

    # ==================== 数值结果输出 ====================

    # ---- 两种方法最优点数值对比 ----
    print("\n" + "=" * 55)
    print("  两种方法最优点数值对比")
    print("=" * 55)
    print(f"{'':20} {'参数化追踪法':>15} {'λ扫描法':>15}")
    print("-" * 55)
    print(f"{'组合预期收益率':<20} {r_opt:>15.4f} {best_point['ret']:>15.4f}")
    print(f"{'组合波动率':<20} {v_opt:>15.4f} {best_point['vol']:>15.4f}")
    print(f"{'夏普比率':<20} {sr_opt:>15.4f} {best_point['sharpe']:>15.4f}")
    print("=" * 55)

    # ---- 参数化法：切线组合权重明细 ----
    print("\n" + "=" * 55)
    print(f"  【参数化法】切线组合最优配置")
    print("=" * 55)
    print(f"{'资产名称':<20} {'最优权重':>10} {'预期收益率':>10}")
    print("-" * 55)
    for i, asset_name in enumerate(assets):
        # 只打印权重 > 0.01% 的资产（过滤数值噪声）
        if w_opt[i] > 1e-4:
            print(f"{asset_name:<20} {w_opt[i]:>10.2%} {mu[i]:>10.4f}")
    print("=" * 55)
    print(f"{'组合预期收益率':<20} {r_opt:>10.4f}")
    print(f"{'组合波动率':<20} {v_opt:>10.4f}")
    print(f"{'夏普比率':<20} {sr_opt:>10.4f}")

    # ---- λ 扫描法：夏普最高点权重明细 ----
    print("\n" + "=" * 55)
    print(f"  【λ扫描法】最优配置（SR最高，λ={best_point['lambda']:.2f}）")
    print("=" * 55)
    print(f"{'资产名称':<20} {'最优权重':>10} {'预期收益率':>10}")
    print("-" * 55)
    for i, asset_name in enumerate(assets):
        if best_point['weights'][i] > 1e-4:
            print(f"{asset_name:<20} {best_point['weights'][i]:>10.2%} {mu[i]:>10.4f}")
    print("=" * 55)
    print(f"{'组合预期收益率':<20} {best_point['ret']:>10.4f}")
    print(f"{'组合波动率':<20} {best_point['vol']:>10.4f}")
    print(f"{'夏普比率':<20} {best_point['sharpe']:>10.4f}")

    # ---- 两种方法权重并排对比 ----
    print("\n" + "=" * 65)
    print(f"  两种方法权重对比")
    print("=" * 65)
    print(f"{'资产名称':<20} {'参数化法':>12} {'λ扫描法':>12} {'差值':>10}")
    print("-" * 65)
    for i, asset_name in enumerate(assets):
        w1   = w_opt[i]                   # 参数化法权重
        w2   = best_point['weights'][i]   # λ 扫描法权重
        diff = w1 - w2                    # 差值（正 = 参数化法更多）
        # 任一方法权重超过 0.01% 即打印
        if w1 > 1e-4 or w2 > 1e-4:
            print(f"{asset_name:<20} {w1:>12.2%} {w2:>12.2%} {diff:>+10.2%}")
    print("=" * 65)

    # ---- 量化两种方法的一致性 ----
    # 在两条前沿的重叠收益率区间内，用插值计算对应波动率的平均偏差
    if len(fv_param) > 1 and len(fv_lambda) > 1:
        # 确定两条前沿在收益率维度上的重叠区间
        ret_min = max(fr_param.min(), fr_lambda.min())
        ret_max = min(fr_param.max(), fr_lambda.max())

        # 在重叠区间内均匀取 20 个收益率点
        common_rets = np.linspace(ret_min, ret_max, 20)

        # 对两条前沿曲线分别做线性插值，得到各收益率水平对应的波动率
        vols_param_interp  = np.interp(common_rets, fr_param,  fv_param)
        vols_lambda_interp = np.interp(common_rets, fr_lambda, fv_lambda)

        # 计算平均绝对偏差
        avg_diff = np.mean(np.abs(vols_param_interp - vols_lambda_interp))

        print(f"\n两条前沿在重叠区间的平均波动率偏差：{avg_diff:.6f}")
        if avg_diff < 0.001:
            print("结论：两种方法高度一致，有效前沿几乎重合")
        elif avg_diff < 0.005:
            print("结论：两种方法基本一致，存在轻微数值差异")
        else:
            print("结论：两种方法存在明显差异，建议检查 λ 扫描范围是否覆盖完整")


# ==================== 8. 主程序入口 ====================

if __name__ == '__main__':
    """
    主程序执行流程：
    1. 调用 load_data() 读取并对齐三张数据表，获得 μ 向量和 Σ 矩阵
    2. 调用 plot_both_frontiers() 执行两种方法的前沿计算和可视化对比

    整个流程是自动化的，只需设置好全局参数（目标账户、文件路径等）即可运行。
    """
    print(f"\n{'=' * 50}")
    print(f"  资产配置优化  |  账户：{target_account}")
    print(f"{'=' * 50}\n")

    # 数据加载与对齐
    assets, mu, cov = load_data(target_account)

    # 有效前沿计算、可视化与数值输出
    plot_both_frontiers(assets, mu, cov)


from constraint_analysis import run_constraint_analysis
run_constraint_analysis(assets, mu, cov, solvency_ratio=1.80)  # 假设当前充足率180%