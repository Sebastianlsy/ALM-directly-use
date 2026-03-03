import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ==================== 0. 中文字体设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 参数设置 ====================
FILE_MAIN = "中间项.xlsx"
FILE_CORR = "相关性矩阵.xlsx"
TARGET_ACCOUNT = '传统账户'
DURATION_GAP_TOLERANCE = 0.5
RF_RATE = 0.02
NUM_PORTFOLIOS = 30000

NAME_MAP = {
    "高等级信用债_3Y": "信用债_3Y",
    "高等级信用债_5Y": "信用债_5Y",
    "高等级信用债_10Y": "信用债_10Y",
}


# ==================== 2. 数据读取与清洗 ====================
def load_data(target_account):
    df_quant = pd.read_excel(FILE_MAIN, sheet_name='量化指标')

    # ---------- 2.1 负债侧（只取久期，不用折现值）----------
    mask_liab = (
        (df_quant['保险账户分类/InsuranceAccountType'] == target_account) &
        (df_quant.iloc[:, 0].str.contains('负债流出\\(资产负债管理-基础\\)', na=False))
    )
    df_liab = df_quant[mask_liab]
    if df_liab.empty:
        raise ValueError(f"未找到账户 [{target_account}] 的负债数据")

    D_liab = df_liab['修正久期/ModifiedDuration'].values[0]
    print(f"[负债] 账户: {target_account} | 久期: {D_liab:.2f}年")

    # ---------- 2.2 资产侧 ----------
    mask_asset = (
        (df_quant['保险账户分类/InsuranceAccountType'] == target_account) &
        (~df_quant.iloc[:, 0].str.contains('负债', na=False)) &
        (df_quant['全价市场价值/DirtyMarketValue'] > 0) &
        (df_quant['资产配置分类三级/SAAAssetTypeLevel3'].notna())
    )
    df_assets = df_quant[mask_asset].copy()
    if df_assets.empty:
        raise ValueError(f"未找到账户 [{target_account}] 的有效资产数据")

    df_assets['asset_std'] = df_assets['资产配置分类三级/SAAAssetTypeLevel3'].map(
        lambda x: NAME_MAP.get(x, x)
    )

    def wavg(g, col):
        w = g['全价市场价值/DirtyMarketValue']
        return np.average(g[col].fillna(0), weights=w)

    df_summary = df_assets.groupby('asset_std').apply(lambda g: pd.Series({
        'mu':       wavg(g, '预期投资收益率/ExpectedReturn'),
        'duration': wavg(g, '修正久期/ModifiedDuration'),
        'mv':       g['全价市场价值/DirtyMarketValue'].sum(),
    }))

    # ---------- 2.3 相关性矩阵 & 波动率 ----------
    df_corr_raw = pd.read_excel(FILE_CORR, sheet_name='相关性', index_col=0)
    df_vol_raw  = pd.read_excel(FILE_CORR, sheet_name='自定义指标')

    drop_items = [c for c in df_corr_raw.index if '负债' in str(c)]
    df_corr_raw = df_corr_raw.drop(index=drop_items, columns=drop_items, errors='ignore')

    arr = df_corr_raw.to_numpy(dtype=float, copy=True)
    mask_nan = np.isnan(arr)
    arr[mask_nan] = arr.T[mask_nan]
    arr = np.nan_to_num(arr, nan=0.0)
    np.fill_diagonal(arr, 1.0)
    corr_matrix = pd.DataFrame(arr, index=df_corr_raw.index, columns=df_corr_raw.columns)

    vol_series = df_vol_raw.set_index(df_vol_raw.columns[0]).iloc[:, 0]
    vol_series = vol_series.drop(index=[i for i in vol_series.index if '负债' in str(i)], errors='ignore')

    # ---------- 2.4 三者取交集 ----------
    common = list(df_summary.index.intersection(corr_matrix.index).intersection(vol_series.index))
    if len(common) == 0:
        raise ValueError("资产名称无法对齐，请检查 NAME_MAP")

    print(f"[资产] 参与优化 {len(common)} 项: {', '.join(common)}")

    mu        = df_summary.loc[common, 'mu'].values
    durations = df_summary.loc[common, 'duration'].values
    sigma     = vol_series.loc[common].values
    corr      = corr_matrix.loc[common, common].values
    cov       = np.diag(sigma) @ corr @ np.diag(sigma)

    eigvals = np.linalg.eigvals(cov)
    if np.any(eigvals < -1e-10):
        print("⚠️  协方差矩阵非正定，已修正")
        cov += (-eigvals.min() + 1e-8) * np.eye(len(common))

    # 诊断：打印各资产久期，帮助判断能否凑到负债久期
    print(f"\n{'资产':<15} {'久期':>8} {'收益率':>10}")
    print("-" * 36)
    for a, d, m in zip(common, durations, mu):
        print(f"{a:<15} {d:>8.2f}年  {m:>9.4f}")
    print(f"\n负债目标久期 = {D_liab:.2f}年")
    print(f"资产最长久期 = {max(durations):.2f}年")
    if max(durations) < D_liab - DURATION_GAP_TOLERANCE:
        print(f"⚠️  资产最长久期({max(durations):.2f}年) < 负债久期({D_liab:.2f}年)-容忍度，久期匹配不可行！")
        print(f"   建议放宽 DURATION_GAP_TOLERANCE 至 {D_liab - max(durations) + 0.1:.1f} 年以上")

    return common, mu, cov, durations, D_liab


# ==================== 3. 狄利克雷采样 ====================
def dirichlet_sampling(mu, cov, durations, D_liab, n=NUM_PORTFOLIOS):
    n_assets = len(mu)
    weights  = np.random.dirichlet([1.0] * n_assets, n)

    port_returns   = weights @ mu
    port_vols      = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov, weights))
    port_sharpe    = (port_returns - RF_RATE) / port_vols
    port_durations = weights @ durations
    duration_gap   = np.abs(port_durations - D_liab)

    return port_returns, port_vols, port_sharpe, duration_gap


# ==================== 4. Scipy 约束优化 ====================
def scipy_optimize(mu, cov, durations, D_liab, with_duration=True):
    n = len(mu)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - RF_RATE) / vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    if with_duration:
        constraints += [
            {'type': 'ineq', 'fun': lambda w: DURATION_GAP_TOLERANCE - (w @ durations - D_liab)},
            {'type': 'ineq', 'fun': lambda w: DURATION_GAP_TOLERANCE + (w @ durations - D_liab)},
        ]

    bounds = [(0, 1)] * n
    best_result = None
    for _ in range(20):
        w0 = np.random.dirichlet([1.0] * n)
        res = minimize(neg_sharpe, w0, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-12, 'maxiter': 1000})
        if res.success and (best_result is None or res.fun < best_result.fun):
            best_result = res

    if best_result is None:
        raise RuntimeError("Scipy 优化失败")

    w_opt = best_result.x
    return w_opt, w_opt @ mu, np.sqrt(w_opt @ cov @ w_opt)


# ==================== 4.5 有效前沿计算 ====================
def calc_efficient_frontier(mu, cov, durations=None, D_liab=None,
                             with_duration=False, n_points=40):
    from scipy.optimize import minimize as _min
    n = len(mu)
    bounds = [(0, 1)] * n
    target_rets = np.linspace(mu.min(), mu.max() * 0.98, n_points)

    frontier_vols, frontier_rets = [], []
    for tr in target_rets:
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=tr: w @ mu - tr},
        ]
        if with_duration and durations is not None:
            cons += [
                {'type': 'ineq', 'fun': lambda w: DURATION_GAP_TOLERANCE - (w @ durations - D_liab)},
                {'type': 'ineq', 'fun': lambda w: DURATION_GAP_TOLERANCE + (w @ durations - D_liab)},
            ]
        best = None
        for _ in range(5):
            w0 = np.random.dirichlet([1.0] * n)
            res = _min(lambda w: np.sqrt(w @ cov @ w), w0, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 1000})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        if best is not None and best.success:
            frontier_vols.append(best.fun)
            frontier_rets.append(tr)

    return np.array(frontier_vols), np.array(frontier_rets)


# ==================== 5. 画图 ====================
def plot_comparison(assets, mu, cov, durations, D_liab):
    rets, vols, sharpes, dgaps = dirichlet_sampling(mu, cov, durations, D_liab)
    mask_dur = dgaps <= DURATION_GAP_TOLERANCE

    print(f"\n久期匹配组合数: {mask_dur.sum()} / {NUM_PORTFOLIOS} ({mask_dur.mean()*100:.1f}%)")

    w_free, r_free, v_free = scipy_optimize(mu, cov, durations, D_liab, with_duration=False)
    w_dur,  r_dur,  v_dur  = scipy_optimize(mu, cov, durations, D_liab, with_duration=True)

    print("计算无约束有效前沿...")
    fv_free, fr_free = calc_efficient_frontier(mu, cov)
    print("计算久期约束有效前沿...")
    fv_dur, fr_dur = calc_efficient_frontier(mu, cov, durations, D_liab, with_duration=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'资产配置优化对比  —  {TARGET_ACCOUNT}', fontsize=15, fontweight='bold')

    # --- 左图：无久期约束 ---
    ax1 = axes[0]
    sc1 = ax1.scatter(vols, rets, c=sharpes, cmap='Blues', alpha=0.3, s=4)
    plt.colorbar(sc1, ax=ax1, label='夏普比率')
    if len(fv_free) > 1:
        min_idx = np.argmin(fv_free)
        ax1.plot(fv_free[min_idx:], fr_free[min_idx:], color='gold', linewidth=2.5, zorder=8, label='有效前沿')
    ax1.scatter(v_free, r_free, color='red', marker='*', s=300, zorder=10,
                label=f'最优点 SR={(r_free - RF_RATE) / v_free:.3f}')
    ax1.set_title('无久期约束', fontsize=13)
    ax1.set_xlabel('组合波动率', fontsize=11)
    ax1.set_ylabel('预期收益率', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- 右图：有久期约束 ---
    ax2 = axes[1]
    ax2.scatter(vols[~mask_dur], rets[~mask_dur],
                color='#FF4444', alpha=0.15, s=4, label='久期不匹配')
    if mask_dur.sum() > 0:
        sc2 = ax2.scatter(vols[mask_dur], rets[mask_dur],
                          c=sharpes[mask_dur], cmap='YlOrRd',
                          alpha=0.75, s=8, zorder=3,
                          label=f'久期匹配 (缺口≤{DURATION_GAP_TOLERANCE}年)')
        plt.colorbar(sc2, ax=ax2, label='夏普比率')
    if len(fv_dur) > 1:
        min_idx2 = np.argmin(fv_dur)
        ax2.plot(fv_dur[min_idx2:], fr_dur[min_idx2:], color='gold', linewidth=2.5,
                 zorder=8, label='有效前沿(久期约束)')
    ax2.scatter(v_dur, r_dur, color='darkred', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.8,
                label=f'最优点 SR={(r_dur - RF_RATE) / v_dur:.3f}')
    ax2.set_title(f'有久期约束 (负债久期={D_liab:.2f}年, 容忍±{DURATION_GAP_TOLERANCE}年)', fontsize=13)
    ax2.set_xlabel('组合波动率', fontsize=11)
    ax2.set_ylabel('预期收益率', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('efficient_frontier_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 efficient_frontier_comparison.png")

    # ---------- 权重对比表 ----------
    print("\n" + "=" * 60)
    print(f"{'资产':<20} {'无约束权重':>12} {'久期约束权重':>14} {'资产久期':>10}")
    print("-" * 60)
    for i, a in enumerate(assets):
        print(f"{a:<20} {w_free[i]:>11.2%} {w_dur[i]:>13.2%} {durations[i]:>9.2f}年")
    print("=" * 60)
    print(f"{'组合久期':<20} {w_free @ durations:>11.2f}年 {w_dur @ durations:>12.2f}年")
    print(f"{'负债久期':<20} {D_liab:>11.2f}年 {D_liab:>12.2f}年")
    print(f"{'久期缺口':<20} {w_free @ durations - D_liab:>+11.2f}年 {w_dur @ durations - D_liab:>+12.2f}年")
    print(f"{'预期收益率':<20} {w_free @ mu:>11.4f} {w_dur @ mu:>13.4f}")
    print(f"{'组合波动率':<20} {v_free:>11.4f} {v_dur:>13.4f}")
    print(f"{'夏普比率':<20} {(w_free @ mu - RF_RATE) / v_free:>11.4f} {(w_dur @ mu - RF_RATE) / v_dur:>13.4f}")


# ==================== 6. 主执行 ====================
if __name__ == '__main__':
    print(f"\n{'=' * 60}")
    print(f"  ALM 资产配置优化  |  账户: {TARGET_ACCOUNT}")
    print(f"{'=' * 60}\n")

    assets, mu, cov, durations, D_liab = load_data(TARGET_ACCOUNT)
    plot_comparison(assets, mu, cov, durations, D_liab)