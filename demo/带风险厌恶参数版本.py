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
file_main      = "中间项.xlsx"
file_corr      = "相关性矩阵.xlsx"
target_account = '万能账户'
rf_rate        = 0.02
num_portfolios = 100000

name_map = {
    "高等级信用债_3Y":  "信用债_3Y",
    "高等级信用债_5Y":  "信用债_5Y",
    "高等级信用债_10Y": "信用债_10Y",
}


# ==================== 2. 数据读取辅助函数 ====================

def read_asset_rows(target_account):
    df_quant = pd.read_excel(file_main, sheet_name='量化指标')
    mask = (
        (df_quant['保险账户分类/InsuranceAccountType'] == target_account) &
        (~df_quant.iloc[:, 0].str.contains('负债', na=False)) &
        (df_quant['全价市场价值/DirtyMarketValue'] > 0) &
        (df_quant['资产配置分类三级/SAAAssetTypeLevel3'].notna())
    )
    df_assets = df_quant[mask].copy()
    if df_assets.empty:
        raise ValueError(f"未找到账户 [{target_account}] 的资产")
    df_assets['asset_std'] = df_assets['资产配置分类三级/SAAAssetTypeLevel3'].map(
        lambda x: name_map.get(x, x)
    )
    return df_assets


def calc_weighted_avg_return(df_assets):
    result = {}
    for asset_name, group in df_assets.groupby('asset_std'):
        market_values   = group['全价市场价值/DirtyMarketValue'].values
        returns         = group['预期投资收益率/ExpectedReturn'].fillna(0).values
        weighted_return = np.average(returns, weights=market_values)
        result[asset_name] = weighted_return
    return pd.DataFrame.from_dict(result, orient='index', columns=['mu'])


def read_corr_matrix():
    df_corr_raw   = pd.read_excel(file_corr, sheet_name='相关性', index_col=0)
    drop_names    = [name for name in df_corr_raw.index if '负债' in str(name)]
    df_corr_raw   = df_corr_raw.drop(index=drop_names, columns=drop_names, errors='ignore')
    arr           = df_corr_raw.to_numpy(dtype=float, copy=True)
    nan_positions = np.isnan(arr)
    arr[nan_positions] = arr.T[nan_positions]
    arr           = np.nan_to_num(arr, nan=0.0)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=df_corr_raw.index, columns=df_corr_raw.columns)


def read_vol_series():
    df_vol     = pd.read_excel(file_corr, sheet_name='自定义指标')
    vol_series = df_vol.set_index(df_vol.columns[0]).iloc[:, 0]
    drop_names = [name for name in vol_series.index if '负债' in str(name)]
    return vol_series.drop(index=drop_names, errors='ignore')


def build_cov_matrix(common, corr_matrix, vol_series):
    sigma       = vol_series.loc[common].values
    corr        = corr_matrix.loc[common, common].values
    cov         = np.diag(sigma) @ corr @ np.diag(sigma)
    eigenvalues = np.linalg.eigvals(cov)
    if np.any(eigenvalues < -1e-10):
        print("警告：协方差矩阵非正定，已修正")
        cov += (-eigenvalues.min() + 1e-8) * np.eye(len(common))
    return cov


def load_data(target_account):
    df_assets   = read_asset_rows(target_account)
    df_summary  = calc_weighted_avg_return(df_assets)
    corr_matrix = read_corr_matrix()
    vol_series  = read_vol_series()

    common = list(
        df_summary.index
        .intersection(corr_matrix.index)
        .intersection(vol_series.index)
    )
    if not common:
        raise ValueError("资产名称三表无法对齐")

    print(f"[资产] 参与优化 {len(common)} 项：{', '.join(common)}")

    mu    = df_summary.loc[common, 'mu'].values
    cov   = build_cov_matrix(common, corr_matrix, vol_series)
    sigma = vol_series.loc[common].values

    print(f"\n{'资产':<18} {'预期收益率':>10} {'波动率':>10}")
    print("-" * 42)
    for a, m, s in zip(common, mu, sigma):
        print(f"{a:<18} {m:>10.4f} {s:>10.4f}")

    return common, mu, cov


# ==================== 3. 狄利克雷随机采样 ====================

def dirichlet_sampling(mu, cov):
    """
    在整个权重单纯形上均匀撒点，生成散点云。
    用于可视化整个可行域，颜色深浅表示夏普比率高低。
    """
    n           = len(mu)
    W           = np.random.dirichlet([1.0] * n, num_portfolios)
    port_ret    = W @ mu
    port_vol    = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))
    port_sharpe = (port_ret - rf_rate) / port_vol
    return port_ret, port_vol, port_sharpe


# ==================== 4. 方法一：参数化追踪法有效前沿 ====================

def calc_frontier_parametric(mu, cov, n_points=60):
    """
    【参数化追踪法】
    固定目标收益率，求最小波动率组合，扫描60个点连成曲线。

    优点：每个点有明确的收益率目标，曲线均匀分布
    缺点：目标收益率接近极值时容易无解，偶尔出现曲线不光滑
    """
    n           = len(mu)
    bounds      = [(0, 1)] * n
    target_rets = np.linspace(mu.min(), mu.max() * 0.98, n_points)

    frontier_vols = []
    frontier_rets = []

    for target_ret in target_rets:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=target_ret: w @ mu - tr},
        ]
        best = None
        for _ in range(5):
            w0  = np.random.dirichlet([1.0] * n)
            res = minimize(
                lambda w: np.sqrt(w @ cov @ w),
                w0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )
            if res.success and (best is None or res.fun < best.fun):
                best = res

        if best is not None and best.success:
            frontier_vols.append(best.fun)
            frontier_rets.append(target_ret)

    frontier_vols = np.array(frontier_vols)
    frontier_rets = np.array(frontier_rets)

    # 只保留最小方差点以上的上半段
    if len(frontier_vols) > 1:
        mi            = np.argmin(frontier_vols)
        frontier_vols = frontier_vols[mi:]
        frontier_rets = frontier_rets[mi:]

    return frontier_vols, frontier_rets


# ==================== 5. 方法二：扫描风险厌恶系数有效前沿 ====================

def calc_frontier_lambda(mu, cov, assets):
    """
    【风险厌恶系数扫描法】
    扫描不同 λ，每个 λ 对应有效前沿上的一个点：
      max  w@μ - (λ/2) × w@Σ@w
      λ 小 -> 偏右上（激进，高收益高风险）
      λ 大 -> 偏左下（保守，低收益低风险）

    优点：每个 λ 都有解，曲线更稳定连续
    缺点：λ 和具体收益率/风险水平不直观对应，需要事后计算

    同时以每个最优点为中心，用集中型狄利克雷撒局部点，
    让散点云聚集在有效前沿附近，而不是均匀铺满整个可行域。
    """
    n = len(mu)

    # 用对数间隔扫描 λ：从 0.1（极激进）到 200（极保守），共60个点
    # 对数间隔让低 λ 区域也有足够的采样密度
    lambda_values = np.logspace(-1, 2.3, 60)

    optimal_points = []  # 存储每个 λ 的最优点信息
    all_vols       = []  # 局部撒点的波动率
    all_rets       = []  # 局部撒点的收益率
    all_sharpes    = []  # 局部撒点的夏普比率

    print("扫描风险厌恶系数，计算有效前沿...")

    for lam in lambda_values:

        # ---- 求当前 λ 下的最优权重 ----
        # 目标函数：最大化均值方差效用 = 收益率 - (λ/2) × 方差
        # 取负变最小化
        def objective(w, lam=lam):
            ret      = w @ mu
            variance = w @ cov @ w   # 注意是方差（不开根号）
            return -(ret - (lam / 2) * variance)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds      = [(0, 1)] * n

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
            continue

        w_opt  = best.x
        r_opt  = w_opt @ mu
        v_opt  = np.sqrt(w_opt @ cov @ w_opt)
        sr_opt = (r_opt - rf_rate) / v_opt

        optimal_points.append({
            'lambda':  lam,
            'weights': w_opt,
            'ret':     r_opt,
            'vol':     v_opt,
            'sharpe':  sr_opt
        })

        # ---- 以最优权重为中心，局部撒点 ----
        # Alpha = w_opt × concentration：权重越大的资产，Alpha越大，撒点越集中在该资产附近
        # +0.1 避免某资产权重为0时 Alpha=0 报错
        concentration = 20
        alpha         = w_opt * concentration + 0.1
        n_local       = 500   # 每个 λ 撒300个局部点

        W_local      = np.random.dirichlet(alpha, n_local)
        local_rets   = W_local @ mu
        local_vols   = np.sqrt(np.einsum('ij,jk,ik->i', W_local, cov, W_local))
        local_sharpes = (local_rets - rf_rate) / local_vols

        all_rets.extend(local_rets)
        all_vols.extend(local_vols)
        all_sharpes.extend(local_sharpes)

    return (
        np.array(all_vols),
        np.array(all_rets),
        np.array(all_sharpes),
        optimal_points
    )


# ==================== 6. 求最大夏普比率最优点 ====================

def find_tangency_portfolio(mu, cov):
    """
    用 SLSQP 精确求解最大夏普比率组合（切线组合）。
    20次多起点策略避免局部最优。
    """
    n = len(mu)

    def neg_sharpe(w):
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


# ==================== 7. 画图：左右两图对比 ====================

def plot_both_frontiers(assets, mu, cov):

    # ---- 准备数据 ----
    print("生成全局散点云...")
    rets_global, vols_global, sharpes_global = dirichlet_sampling(mu, cov)

    print("计算参数化有效前沿...")
    fv_param, fr_param = calc_frontier_parametric(mu, cov)

    print("求切线组合...")
    w_opt, r_opt, v_opt = find_tangency_portfolio(mu, cov)
    sr_opt = (r_opt - rf_rate) / v_opt

    print("扫描风险厌恶系数...")
    vols_local, rets_local, sharpes_local, optimal_points = calc_frontier_lambda(mu, cov, assets)

    fv_lambda = np.array([p['vol'] for p in optimal_points])
    fr_lambda = np.array([p['ret'] for p in optimal_points])
    best_point = max(optimal_points, key=lambda p: p['sharpe'])

    # ---- 三图布局 ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f'有效前沿方法对比  —  {target_account}',
                 fontsize=15, fontweight='bold')

    # ==================== 左图：参数化追踪法 ====================
    ax1.set_title('参数化追踪法\n固定收益率 → 最小波动率', fontsize=11)

    sc1 = ax1.scatter(vols_global, rets_global, c=sharpes_global,
                      cmap='Blues', alpha=0.3, s=5)
    plt.colorbar(sc1, ax=ax1, label='夏普比率')

    if len(fv_param) > 1:
        ax1.plot(fv_param, fr_param, color='gold', linewidth=2.5,
                 zorder=8, label='有效前沿')

    ax1.scatter(v_opt, r_opt, color='red', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.5,
                label=f'最优点 SR={sr_opt:.3f}')

    ax1.set_xlabel('组合波动率', fontsize=11)
    ax1.set_ylabel('预期收益率', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ==================== 中图：λ 扫描法 ====================
    ax2.set_title('风险厌恶扫描法\n扫描 λ → 最大化均值方差效用', fontsize=11)

    sc2 = ax2.scatter(vols_local, rets_local, c=sharpes_local,
                      cmap='Blues', alpha=0.4, s=5)
    plt.colorbar(sc2, ax=ax2, label='夏普比率')

    if len(fv_lambda) > 1:
        ax2.plot(fv_lambda, fr_lambda, color='gold', linewidth=2.5,
                 zorder=8, label='有效前沿')

    # 标注关键 λ 值
    label_lambdas = [0.1, 1, 5, 20, 100]
    for p in optimal_points:
        if any(abs(p['lambda'] - lam) / lam < 0.15 for lam in label_lambdas):
            ax2.annotate(
                f"λ={p['lambda']:.1f}",
                xy=(p['vol'], p['ret']),
                xytext=(p['vol'] + 0.003, p['ret']),
                fontsize=8, color='dimgray',
                arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.8)
            )

    ax2.scatter(best_point['vol'], best_point['ret'],
                color='red', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.5,
                label=f"最优点 SR={best_point['sharpe']:.3f}")

    ax2.set_xlabel('组合波动率', fontsize=11)
    ax2.set_ylabel('预期收益率', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ==================== 右图：两条前沿叠加对比 ====================
    # 这张图是关键：把两条曲线画在同一坐标系，直接看差异
    ax3.set_title('两种方法有效前沿对比\n（曲线越接近说明两种方法越一致）', fontsize=11)

    # 参数化前沿：金色实线
    if len(fv_param) > 1:
        ax3.plot(fv_param, fr_param, color='gold', linewidth=2.5,
                 zorder=8, label='参数化追踪法', linestyle='-')

    # λ 扫描前沿：橙红色虚线
    if len(fv_lambda) > 1:
        ax3.plot(fv_lambda, fr_lambda, color='tomato', linewidth=2.5,
                 zorder=9, label='λ 扫描法', linestyle='--')

    # 两个最优点
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

    plt.tight_layout()
    plt.savefig('frontier_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 frontier_comparison.png")

    # ---- 数值对比表 ----
    print("\n" + "=" * 55)
    print("  两种方法最优点数值对比")
    print("=" * 55)
    print(f"{'':20} {'参数化追踪法':>15} {'λ扫描法':>15}")
    print("-" * 55)
    print(f"{'组合预期收益率':<20} {r_opt:>15.4f} {best_point['ret']:>15.4f}")
    print(f"{'组合波动率':<20} {v_opt:>15.4f} {best_point['vol']:>15.4f}")
    print(f"{'夏普比率':<20} {sr_opt:>15.4f} {best_point['sharpe']:>15.4f}")
    print("=" * 55)

    # ---- 参数化法：切线组合权重 ----
    print("\n" + "=" * 55)
    print(f"  【参数化法】切线组合最优配置")
    print("=" * 55)
    print(f"{'资产名称':<20} {'最优权重':>10} {'预期收益率':>10}")
    print("-" * 55)
    for i, asset_name in enumerate(assets):
        if w_opt[i] > 1e-4:
            print(f"{asset_name:<20} {w_opt[i]:>10.2%} {mu[i]:>10.4f}")
    print("=" * 55)
    print(f"{'组合预期收益率':<20} {r_opt:>10.4f}")
    print(f"{'组合波动率':<20} {v_opt:>10.4f}")
    print(f"{'夏普比率':<20} {sr_opt:>10.4f}")

    # ---- λ扫描法：夏普最高点权重 ----
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
        w1 = w_opt[i]
        w2 = best_point['weights'][i]
        diff = w1 - w2
        # 两种方法任意一个权重超过0.01%就打印
        if w1 > 1e-4 or w2 > 1e-4:
            print(f"{asset_name:<20} {w1:>12.2%} {w2:>12.2%} {diff:>+10.2%}")
    print("=" * 65)

    # 两条前沿曲线在重叠收益率区间内的平均偏差
    # 用于量化两种方法的一致性
    if len(fv_param) > 1 and len(fv_lambda) > 1:
        ret_min = max(fr_param.min(), fr_lambda.min())
        ret_max = min(fr_param.max(), fr_lambda.max())

        # 在重叠区间插值对比
        common_rets = np.linspace(ret_min, ret_max, 20)
        vols_param_interp  = np.interp(common_rets, fr_param,  fv_param)
        vols_lambda_interp = np.interp(common_rets, fr_lambda, fv_lambda)
        avg_diff = np.mean(np.abs(vols_param_interp - vols_lambda_interp))

        print(f"\n两条前沿在重叠区间的平均波动率偏差：{avg_diff:.6f}")
        if avg_diff < 0.001:
            print("结论：两种方法高度一致，有效前沿几乎重合")
        elif avg_diff < 0.005:
            print("结论：两种方法基本一致，存在轻微数值差异")
        else:
            print("结论：两种方法存在明显差异，建议检查 λ 扫描范围是否覆盖完整")


# ==================== 8. 主程序 ====================

if __name__ == '__main__':
    print(f"\n{'=' * 50}")
    print(f"  资产配置优化  |  账户：{target_account}")
    print(f"{'=' * 50}\n")

    assets, mu, cov = load_data(target_account)
    plot_both_frontiers(assets, mu, cov)
