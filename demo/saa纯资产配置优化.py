import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ==================== 0. 中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 全局参数 ====================
file_main      = "中间项.xlsx"      # 资产量化指标（预期收益率、市值、久期等）
file_corr      = "相关性矩阵.xlsx"  # 相关性矩阵（资产间相关系数和各资产波动率）
target_account = '万能账户'         # 目标账户，可切换为：传统账户 / 分红账户

# 无风险利率：通常取10年期国债收益率，用于计算夏普比率
# 夏普比率 = (组合收益率 - 无风险利率) / 组合波动率，衡量单位风险的超额收益(越高越好)
rf_rate = 0.02

# 随机采样数
num_portfolios = 200000

# 资产名称映射：中间项命名 -> 相关性矩阵命名（只需填写不一致的部分）
name_map = {
    "高等级信用债_3Y":  "信用债_3Y",
    "高等级信用债_5Y":  "信用债_5Y",
    "高等级信用债_10Y": "信用债_10Y",
}


# ==================== 2. 数据处理功能函数 ====================

def read_asset_rows(target_account):
    """
    从中间项读取目标账户的资产行，返回原始 DataFrame。

    筛选逻辑：
      - 账户类型匹配
      - 排除负债行（负债不参与资产端优化）
      - 全价市值 > 0（排除零持仓行）
      - 有三级分类
    """
    df_quant = pd.read_excel(file_main, sheet_name='量化指标')

    mask = (
        (df_quant['保险账户分类/InsuranceAccountType'] == target_account) &
        (~df_quant.iloc[:, 0].str.contains('负债', na=False)) &
        (df_quant['全价市场价值/DirtyMarketValue'] > 0) &
        (df_quant['资产配置分类三级/SAAAssetTypeLevel3'].notna())
    )
    df_assets = df_quant[mask].copy()

    if df_assets.empty:
        raise ValueError(f"未找到账户 [{target_account}] 的资产，请检查账户名称")

    # 统一资产命名，对齐相关性矩阵
    df_assets['asset_std'] = df_assets['资产配置分类三级/SAAAssetTypeLevel3'].map(
        lambda x: name_map.get(x, x)
    )
    return df_assets


def calc_weighted_avg_return(df_assets):
    """
    对同一资产类别的多笔持仓，用全价市值做加权平均收益率，汇总为一行。
    同一资产类别（如"配置盘政府债"）可能买了多只不同债券，
    每只债券的预期收益率略有差异，需要加权平均为一个代表性数字，
    才能在优化时把它们当作一类资产统一处理。

    返回：以 asset_std 为索引，包含 'mu'（加权平均收益率）列的 DataFrame。
    """
    result = {}

    for asset_name, group in df_assets.groupby('asset_std'):
        # 取出该类资产各笔持仓的市值和收益率
        market_values = group['全价市场价值/DirtyMarketValue'].values
        returns = group['预期投资收益率/ExpectedReturn'].fillna(0).values

        # 加权平均收益率 = sum(市值_i × 收益率_i) / sum(市值_i)
        weighted_return = np.average(returns, weights=market_values)
        result[asset_name] = weighted_return

    # 转为 DataFrame，方便后续按名称索引
    df_summary = pd.DataFrame.from_dict(result, orient='index', columns=['mu'])
    return df_summary


def read_corr_matrix():
    """
    从相关性矩阵文件读取相关系数，并将下三角补全为完整对称阵。
    相关性矩阵 ρ_ij = ρ_ji，系统导出时只存了下三角（节省空间），上三角为 NaN，需要手动补全。

    步骤：
      1. 用转置填充上三角的 NaN（ρ_ij = ρ_ji）
      2. 若两边都是 NaN（该对资产完全无数据），填 0（假设不相关）
      3. 对角线强制为 1（资产与自身相关性=1）

    完整的相关系数 DataFrame，行列均为资产名。
    """
    df_corr_raw = pd.read_excel(file_corr, sheet_name='相关性', index_col=0)

    # 剔除负债行列（资产优化用不到）
    drop_names = [name for name in df_corr_raw.index if '负债' in str(name)]
    df_corr_raw = df_corr_raw.drop(index=drop_names, columns=drop_names, errors='ignore')

    # 转为 numpy 数组方便操作
    arr = df_corr_raw.to_numpy(dtype=float, copy=True)

    # 1：找到 NaN 的位置，用对称位置（转置）的值填充
    nan_positions = np.isnan(arr)
    arr[nan_positions] = arr.T[nan_positions]

    # 2：仍然是 NaN 的位置（两边都缺失），填 0
    arr = np.nan_to_num(arr, nan=0.0)

    # 3：对角线强制为 1
    np.fill_diagonal(arr, 1.0)

    # 还原为带行列标签的 DataFrame
    corr_matrix = pd.DataFrame(arr, index=df_corr_raw.index, columns=df_corr_raw.columns)
    return corr_matrix


def read_vol_series():
    """
    从相关性矩阵文件读取各资产的年化波动率。

    自定义指标 Sheet 格式：第一列为资产名，第二列为波动率值。

    以资产名为索引的波动率 Series。
    """
    df_vol = pd.read_excel(file_corr, sheet_name='自定义指标')

    # 第一列设为索引（资产名），取第二列（波动率）
    vol_series = df_vol.set_index(df_vol.columns[0]).iloc[:, 0]

    # 剔除负债行
    drop_names = [name for name in vol_series.index if '负债' in str(name)]
    vol_series = vol_series.drop(index=drop_names, errors='ignore')

    return vol_series


def build_cov_matrix(common, corr_matrix, vol_series):
    """
    用波动率向量和相关系数矩阵构造协方差矩阵，并做正定性修正。

    【协方差矩阵公式】
    cov_ij = σ_i × ρ_ij × σ_j
    矩阵形式：cov = D @ corr @ D
    其中 D = diag(σ_1, σ_2, ..., σ_n) 是以各资产波动率为对角元素的方阵

    【为什么需要正定性修正】
    scipy 优化器要求协方差矩阵正定（所有特征值 > 0），否则 sqrt(w @ cov @ w) 可能出现虚数或数值不稳定。
    若相关性矩阵因数据缺失被填了 0，可能导致矩阵非正定。
    修正方法：给矩阵加一个极小的单位矩阵扰动，
             使最小特征值从负数变为正数，对实际结果影响极小。

    返回：正定的协方差矩阵，形状 (n, n)。
    """
    sigma = vol_series.loc[common].values           # 波动率向量，形状 (n,)
    corr = corr_matrix.loc[common, common].values  # 相关系数矩阵，形状 (n, n)

    # 构造对角波动率矩阵并计算协方差矩阵
    D   = np.diag(sigma)
    cov = D @ corr @ D

    # 正定性检验：计算所有特征值，看是否有负值
    eigenvalues = np.linalg.eigvals(cov)
    if np.any(eigenvalues < -1e-10):
        print("警告：协方差矩阵非正定，已做最小特征值修正")
        # 加扰动：让最小特征值变为正的极小量
        perturbation = -eigenvalues.min() + 1e-8
        cov = cov + perturbation * np.eye(len(common))

    return cov


# ==================== 3. 主数据读取函数 ====================

def load_data(target_account):
    """
    调用上面各辅助函数，完成数据读取和预处理，返回优化所需的三个输入。

    【均值-方差模型 (Markowitz, 1952) 所需输入】
      mu：预期收益率向量，每项资产的预期年化收益率
      cov：协方差矩阵，对角线 = 各资产方差，非对角线 = 资产间协方差

    【三表取交集】
    收益率、相关性矩阵、波动率来自不同数据源，资产名称可能不完全一致。
    只保留三个数据源中都有数据的资产，缺任何一项都无法参与优化。

    return：
      common  参与优化的资产名列表
      mu      预期收益率向量，形状 (n,)
      cov     协方差矩阵，形状 (n, n)
    """
    # 第一步：读取各数据源
    df_assets   = read_asset_rows(target_account)
    df_summary  = calc_weighted_avg_return(df_assets)
    corr_matrix = read_corr_matrix()
    vol_series  = read_vol_series()

    # 第二步：三表取名称交集
    common = list(
        df_summary.index
        .intersection(corr_matrix.index)
        .intersection(vol_series.index)
    )
    if not common:
        raise ValueError("资产名称三表无法对齐，请检查 name_map 或相关性矩阵命名")

    print(f"[资产] 参与优化 {len(common)} 项：{', '.join(common)}")

    # 第三步：提取对齐后的数据
    mu  = df_summary.loc[common, 'mu'].values   # 收益率向量
    cov = build_cov_matrix(common, corr_matrix, vol_series)  # 协方差矩阵

    # 打印各资产数据供核验
    sigma = vol_series.loc[common].values
    print(f"\n{'资产':<18} {'预期收益率':>10} {'波动率':>10}")
    print("-" * 42)
    for a, m, s in zip(common, mu, sigma):
        print(f"{a:<18} {m:>10.4f} {s:>10.4f}")

    return common, mu, cov


# ==================== 4. 狄利克雷随机采样 ====================

def dirichlet_sampling(mu, cov):
    """
    随机生成大量投资组合

    【为什么用狄利克雷分布】
    直接均匀随机生成权重再归一化，在高维情况下样本会集中在单纯形中心，
    边缘区域（极端配置，如全仓某一资产）几乎采样不到。
    狄利克雷分布（Alpha 全为 1）在权重单纯形上均匀分布，
    能覆盖从"全仓单一资产"到"均匀分散"的所有情形。
    """
    n = len(mu)

    # 生成随机权重矩阵，形状 (num_portfolios, n)，每行和为 1
    W = np.random.dirichlet([1.0] * n, num_portfolios)

    # 逐组合计算收益率（直接矩阵乘法）
    port_ret = W @ mu

    # 逐组合计算波动率
    # einsum('ij,jk,ik->i') 等价于对每行 w 计算 w @ cov @ w，向量化批量运算
    port_vol = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))

    # 逐组合计算夏普比率
    port_sharpe = (port_ret - rf_rate) / port_vol

    return port_ret, port_vol, port_sharpe


# ==================== 5. 求最大夏普比率最优组合 ====================

def scipy_optimize(mu, cov):
    """
    用 SLSQP 精确求解最大夏普比率组合（切线组合）。

    【切线组合 (Tangency Portfolio)】
    从无风险利率点向有效前沿做切线，切点对应的组合即为切线组合。
    该组合夏普比率最高，是风险资产的最优配置方案。

    【SLSQP 算法】
    序列二次规划（Sequential Least Squares Programming）：
    每次迭代将非线性问题近似为二次规划，迭代收敛到最优解。
    适合带等式/不等式约束的中小规模优化（资产数 < 100）。

    【多起点策略】
    运行 20 次，每次用不同随机初始权重，取最优结果，
    避免因初始点差导致陷入局部最优。
    """
    n = len(mu)

    # 目标函数：取负夏普（scipy 只能最小化，取负后等价于最大化夏普）
    def neg_sharpe(w):
        portfolio_return = w @ mu
        portfolio_vol = np.sqrt(w @ cov @ w)
        sharpe_ratio = (portfolio_return - rf_rate) / portfolio_vol
        return -sharpe_ratio

    # 等式约束：权重之和 = 1（全仓，不持有现金之外的无风险资产）
    sum_to_one = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    constraints = [sum_to_one]

    # 各资产权重范围 [0, 1]
    bounds = [(0, 1)] * n

    best_result = None
    for _ in range(20):
        # 随机生成初始权重，满足非负且和为 1
        w0 = np.random.dirichlet([1.0] * n)

        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        # 保留成功收敛且夏普更高（目标函数值更小）的结果
        if result.success:
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    if best_result is None:
        raise RuntimeError("优化失败，请检查数据是否有误（如收益率全为0）")

    w_opt = best_result.x
    r_opt = w_opt @ mu
    v_opt = np.sqrt(w_opt @ cov @ w_opt)

    return w_opt, r_opt, v_opt


# ==================== 6. 计算有效前沿 ====================

def calc_efficient_frontier(mu, cov, n_points=40):
    """
    计算有效前沿曲线：在不同目标收益率下求最小波动率组合，连成曲线。

    【有效前沿 (Efficient Frontier)】
    有效前沿上的每个点都是"给定收益率下波动率最低"的组合，代表了在当前资产集合和约束条件下能达到的最优风险-收益权衡。

    【参数化追踪法】
    均匀取 n_points 个目标收益率，对每个目标收益率求解：
      min   sqrt(w @ cov @ w)       最小化波动率
      s.t.  Σw_i = 1                权重和为1
            w @ μ = 目标收益率      达到指定收益
            w_i >= 0               不做空
    """
    n = len(mu)
    bounds = [(0, 1)] * n

    # 扫描范围：最低到最高收益率的 98%（留余量避免无解）
    # 在最低收益和最高收益之间均匀取 n_points个数
    target_rets = np.linspace(mu.min(), mu.max() * 0.98, n_points)

    frontier_vols = []  # 各目标收益率对应的最优波动率
    frontier_rets = []  # 各目标收益率

    for target_ret in target_rets:
        # 对当前目标收益率，求最小波动率组合
        sum_to_one   = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        hit_target   = {'type': 'eq', 'fun': lambda w, tr=target_ret: w @ mu - tr}
        # 注意 tr=target_ret：lambda 默认参数捕获循环变量，避免所有迭代用同一个值
        constraints  = [sum_to_one, hit_target]

        best_result = None
        for _ in range(5):  # 5 次随机起点，平衡精度和速度
            w0 = np.random.dirichlet([1.0] * n)

            result = minimize(
                lambda w: np.sqrt(w @ cov @ w),  # 最小化波动率
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )

            if result.success:
                if best_result is None or result.fun < best_result.fun: # .fun 目标函数的最优解
                    best_result = result

        if best_result is not None and best_result.success:
            frontier_vols.append(best_result.fun)
            frontier_rets.append(target_ret)

    frontier_vols = np.array(frontier_vols)
    frontier_rets = np.array(frontier_rets)

    # 只保留最小方差点以上的上半段（有效前沿）
    if len(frontier_vols) > 1:
        min_var_idx   = np.argmin(frontier_vols)
        frontier_vols = frontier_vols[min_var_idx:]
        frontier_rets = frontier_rets[min_var_idx:]

    return frontier_vols, frontier_rets


# ==================== 7. 画图与结果输出 ====================

def plot_result(assets, mu, cov):
    """
    绘制资产配置全景图，叠加三层信息：
      1. 蓝色散点云：狄利克雷随机采样的可行组合（颜色=夏普比率）
      2. 金色曲线：有效前沿（最优风险-收益边界）
      3. 红色五角星：最大夏普比率最优点（切线组合）
    """
    # 生成散点云
    rets, vols, sharpes = dirichlet_sampling(mu, cov)

    # 求最优点
    w_opt, r_opt, v_opt = scipy_optimize(mu, cov)
    sr_opt = (r_opt - rf_rate) / v_opt

    # 计算有效前沿
    print("计算有效前沿，请稍候...")
    frontier_vols, frontier_rets = calc_efficient_frontier(mu, cov)

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f'资产配置优化  —  {target_account}', fontsize=15, fontweight='bold')

    # 散点云：颜色深浅表示夏普比率高低
    sc = ax.scatter(vols, rets, c=sharpes, cmap='Blues', alpha=0.3, s=5)
    plt.colorbar(sc, ax=ax, label='夏普比率')

    # 有效前沿：金色线，画在散点上方
    if len(frontier_vols) > 1:
        ax.plot(frontier_vols, frontier_rets,
                color='gold', linewidth=2.5, zorder=8, label='有效前沿')

    # 最优点：红色五角星
    ax.scatter(v_opt, r_opt, color='red', marker='*', s=350, zorder=10,
               edgecolors='white', linewidths=0.5,
               label=f'最优点  收益={r_opt:.4f}  波动={v_opt:.4f}  SR={sr_opt:.3f}')

    ax.set_xlabel('组合波动率', fontsize=12)
    ax.set_ylabel('预期收益率', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('asset_allocation_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 asset_allocation_result.png")

    # ---- 最优权重表 ----
    print("\n" + "=" * 50)
    print(f"  最优资产配置方案  —  {target_account}")
    print("=" * 50)
    print(f"{'资产名称':<20} {'最优权重':>10} {'预期收益率':>10}")
    print("-" * 50)
    for i, asset_name in enumerate(assets):
        if w_opt[i] > 1e-4:  # 过滤掉优化器产生的数值噪声（权重极小视为0）
            print(f"{asset_name:<20} {w_opt[i]:>10.2%} {mu[i]:>10.4f}")
    print("=" * 50)
    print(f"{'组合预期收益率':<20} {r_opt:>10.4f}")
    print(f"{'组合波动率':<20} {v_opt:>10.4f}")
    print(f"{'夏普比率':<20} {sr_opt:>10.4f}")


# ==================== 8. 主程序 ====================
if __name__ == '__main__':
    print(f"\n{'=' * 50}")
    print(f"  资产配置优化  |  账户：{target_account}")
    print(f"{'=' * 50}\n")

    assets, mu, cov = load_data(target_account)
    plot_result(assets, mu, cov)