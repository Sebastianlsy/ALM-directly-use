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
file_corr      = "约束条件.xlsx"
target_account = '万能账户'
rf_rate        = 0.02
num_portfolios = 100000

# 风险厌恶图：选定一个固定 λ 用于集中撒点
# （用户可根据需要调整此值）
lambda_for_scatter = 5.0

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
    vol_series = df_vol.set_index(df_vol.columns[0])['波动率/Volatility']
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


# ==================== 3. 约束条件读取 ====================

def build_constraints(assets):
    df_c   = pd.read_excel(file_corr, sheet_name='资产约束')
    df_tag = pd.read_excel(file_corr, sheet_name='自定义标签约束')
    n         = len(assets)
    asset_idx = {a: i for i, a in enumerate(assets)}

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    for level1, group in df_c.groupby('第一级分类/Level1'):
        lo      = group['第一级分类大于/Level1GreaterThan'].iloc[0]
        hi      = group['第一级分类小于/Level1LessThan'].iloc[0]
        members = [a for a in group['资产组合名称/PortfolioName'] if a in asset_idx]
        if not members:
            continue
        idx = [asset_idx[a] for a in members]
        constraints.append({'type': 'ineq', 'fun': lambda w, i=idx, lo=lo: np.sum(w[i]) - lo})
        constraints.append({'type': 'ineq', 'fun': lambda w, i=idx, hi=hi: hi - np.sum(w[i])})

    for _, row in df_c.iterrows():
        asset = row['资产组合名称/PortfolioName']
        if asset not in asset_idx:
            continue
        i    = asset_idx[asset]
        l2lo = row['第二级分类大于/Level2GreaterThan']
        l2hi = row['第二级分类小于/Level2LessThan']
        if l2hi < 1.0:
            constraints.append({'type': 'ineq', 'fun': lambda w, i=i, hi=l2hi: hi - w[i]})
        if l2lo > 0:
            constraints.append({'type': 'ineq', 'fun': lambda w, i=i, lo=l2lo: w[i] - lo})

    if '泛类权益' in df_tag['标签/TagName'].values:
        row0      = df_tag[df_tag['标签/TagName'] == '泛类权益'].iloc[0]
        eq_lo     = row0['大于/GreaterThan']
        eq_hi     = row0['小于/LessThan']
        eq_assets = df_tag[df_tag['标签/TagName'] == '泛类权益']['资产组合名称/PortfolioName'].tolist()
        eq_idx    = [asset_idx[a] for a in eq_assets if a in asset_idx]
        constraints.append({'type': 'ineq', 'fun': lambda w, i=eq_idx, lo=eq_lo: np.sum(w[i]) - lo})
        constraints.append({'type': 'ineq', 'fun': lambda w, i=eq_idx, hi=eq_hi: hi - np.sum(w[i])})

    return constraints


def _extra_cons(constraints, n):
    dummy = np.ones(n) / n
    return [c for c in constraints
            if not (c['type'] == 'eq' and abs(c['fun'](dummy)) < 1e-6)]


# ==================== 4. 约束检查函数 ====================

def check_constraints(w, constraints, tol=1e-6):
    """
    检查单个权重向量 w 是否满足所有约束条件。
    eq 约束：|fun(w)| <= tol
    ineq 约束：fun(w) >= -tol
    """
    for c in constraints:
        val = c['fun'](w)
        if c['type'] == 'eq' and abs(val) > tol:
            return False
        if c['type'] == 'ineq' and val < -tol:
            return False
    return True


def check_constraints_batch(W, constraints, tol=1e-6):
    """
    批量检查，返回布尔掩码数组。
    对大量点逐行检查约束，只保留满足所有约束的点。
    """
    n_samples = W.shape[0]
    mask = np.ones(n_samples, dtype=bool)
    for i in range(n_samples):
        if not check_constraints(W[i], constraints, tol):
            mask[i] = False
    return mask


# ==================== 5. 狄利克雷随机采样（支持约束过滤）====================

def dirichlet_sampling(mu, cov, constraints=None):
    """
    在整个权重单纯形上均匀撒点。
    若传入 constraints，则只保留满足约束的点。
    为保证有足够多的可行点，采用多批次采样策略。
    """
    n = len(mu)

    if constraints is None:
        # 无约束：直接全域均匀采样
        W = np.random.dirichlet([1.0] * n, num_portfolios)
        port_ret    = W @ mu
        port_vol    = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))
        port_sharpe = (port_ret - rf_rate) / port_vol
        return port_ret, port_vol, port_sharpe
    else:
        # 有约束：采样后过滤，分批进行直到收集够点
        target_count = num_portfolios
        batch_size   = num_portfolios  # 每批采样数量
        max_batches  = 20              # 最多采样批数，防止死循环

        W_feasible = []
        for batch_i in range(max_batches):
            W_batch = np.random.dirichlet([1.0] * n, batch_size)
            mask    = check_constraints_batch(W_batch, constraints)
            W_feasible.append(W_batch[mask])

            total = sum(len(w) for w in W_feasible)
            print(f"    批次 {batch_i+1}: 本批采样 {batch_size}, "
                  f"可行 {mask.sum()}, 累计可行 {total}")
            if total >= target_count:
                break

        W_all = np.vstack(W_feasible)[:target_count]
        print(f"    最终可行散点数: {len(W_all)}")

        port_ret    = W_all @ mu
        port_vol    = np.sqrt(np.einsum('ij,jk,ik->i', W_all, cov, W_all))
        port_sharpe = (port_ret - rf_rate) / port_vol
        return port_ret, port_vol, port_sharpe


# ==================== 6. 参数化追踪法有效前沿（延长范围）====================

def calc_frontier_parametric(mu, cov, n_points=80, constraints=None):
    """
    固定目标收益率，求最小波动率组合。
    修改：将目标收益率范围从 mu.max()*0.98 扩展到 mu.max()*1.05，
    并增加点数到80，使前沿更完整。
    若某个目标收益率无法求解（特别是超出可行域时），则自动跳过。
    """
    n           = len(mu)
    bounds      = [(0, 1)] * n
    # 【修改】延长上界到 mu.max() * 1.05，让前沿尽可能向高收益端延伸
    target_rets = np.linspace(mu.min() * 0.95, mu.max() * 1.05, n_points)
    extra       = _extra_cons(constraints, n) if constraints else []

    frontier_vols = []
    frontier_rets = []

    for target_ret in target_rets:
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=target_ret: w @ mu - tr},
        ] + extra

        best = None
        for _ in range(8):  # 增加多起点次数
            w0  = np.random.dirichlet([1.0] * n)
            res = minimize(
                lambda w: np.sqrt(w @ cov @ w),
                w0, method='SLSQP', bounds=bounds, constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )
            if res.success and (best is None or res.fun < best.fun):
                best = res

        if best is not None and best.success:
            frontier_vols.append(best.fun)
            frontier_rets.append(target_ret)

    frontier_vols = np.array(frontier_vols)
    frontier_rets = np.array(frontier_rets)

    # 只保留最小波动率点之上的部分（上半前沿）
    if len(frontier_vols) > 1:
        mi            = np.argmin(frontier_vols)
        frontier_vols = frontier_vols[mi:]
        frontier_rets = frontier_rets[mi:]

    return frontier_vols, frontier_rets


# ==================== 7. 风险厌恶系数扫描法 ====================

def calc_frontier_lambda(mu, cov, assets, constraints=None):
    """
    扫描不同 λ，每个 λ 对应一个均值方差效用最优点，形成有效前沿。
    返回前沿点和所有最优点信息（不在此处撒散点）。
    """
    n             = len(mu)
    lambda_values = np.logspace(-1, 2.3, 80)  # 增加扫描密度
    extra         = _extra_cons(constraints, n) if constraints else []
    optimal_points = []

    print("  扫描风险厌恶系数...")

    for lam in lambda_values:

        def objective(w, lam=lam):
            return -(w @ mu - (lam / 2) * (w @ cov @ w))

        cons   = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + extra
        bounds = [(0, 1)] * n

        best = None
        for _ in range(20):
            w0  = np.random.dirichlet([1.0] * n)
            res = minimize(objective, w0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'ftol': 1e-12, 'maxiter': 1000})
            if res.success and (best is None or res.fun < best.fun):
                best = res

        if best is None:
            continue

        w_opt  = best.x
        r_opt  = w_opt @ mu
        v_opt  = np.sqrt(w_opt @ cov @ w_opt)
        sr_opt = (r_opt - rf_rate) / v_opt

        optimal_points.append({
            'lambda': lam, 'weights': w_opt,
            'ret': r_opt, 'vol': v_opt, 'sharpe': sr_opt
        })

    return optimal_points


def lambda_focused_sampling(mu, cov, w_center, num_samples=100000, concentration=20.0):
    """
    【修正后的风险厌恶散点逻辑】
    给定一个固定 λ 下的最优权重 w_center，
    用 w_center * concentration 作为 Dirichlet 分布参数进行撒点。
    这样散点会集中在最优点附近，体现该 λ 下的局部风险-收益特征。

    参数：
      w_center      : 固定 λ 下的最优权重向量
      num_samples   : 撒点总数
      concentration : 集中度（默认20，即 alpha = w * 20）
    """
    # alpha = w_center * concentration，加一个小量避免 alpha=0
    alpha = w_center * concentration + 0.01
    W     = np.random.dirichlet(alpha, num_samples)

    port_ret    = W @ mu
    port_vol    = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))
    port_sharpe = (port_ret - rf_rate) / port_vol
    return port_ret, port_vol, port_sharpe


# ==================== 8. 求最大夏普比率组合 ====================

def find_tangency_portfolio(mu, cov, constraints=None):
    n      = len(mu)
    extra  = _extra_cons(constraints, n) if constraints else []
    cons   = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + extra
    bounds = [(0, 1)] * n

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf_rate) / vol

    best = None
    for _ in range(20):
        w0  = np.random.dirichlet([1.0] * n)
        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'ftol': 1e-12, 'maxiter': 1000})
        if res.success and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        raise RuntimeError("最优点求解失败")
    w_opt = best.x
    return w_opt, w_opt @ mu, np.sqrt(w_opt @ cov @ w_opt)


# ==================== 9. 结果打印函数 ====================

def print_result(label, assets, mu, w_opt, r_opt, v_opt, w_ref=None, label_ref=None):
    sr = (r_opt - rf_rate) / v_opt
    print(f"\n{'=' * 60}")
    print(f"  【{label}】")
    print(f"{'=' * 60}")
    print(f"  组合预期收益率：{r_opt:.4f}   波动率：{v_opt:.4f}   夏普比率：{sr:.4f}")
    print(f"\n  {'资产名称':<20} {'权重':>10} {'预期收益率':>12}")
    print(f"  {'-' * 44}")
    for i, a in enumerate(assets):
        if w_opt[i] > 1e-4:
            print(f"  {a:<20} {w_opt[i]:>10.2%} {mu[i]:>12.4f}")

    if w_ref is not None and label_ref is not None:
        print(f"\n  与【{label_ref}】的权重差异：")
        print(f"  {'资产名称':<20} {'本方法':>10} {label_ref:>12} {'差值':>10}")
        print(f"  {'-' * 55}")
        for i, a in enumerate(assets):
            w1, w2 = w_opt[i], w_ref[i]
            if w1 > 1e-4 or w2 > 1e-4:
                print(f"  {a:<20} {w1:>10.2%} {w2:>12.2%} {w1 - w2:>+10.2%}")


def print_three_way_compare(assets, mu,
                             w_unc, r_unc, v_unc,
                             w_con, r_con, v_con,
                             w_lam, r_lam, v_lam):
    sr_unc = (r_unc - rf_rate) / v_unc
    sr_con = (r_con - rf_rate) / v_con
    sr_lam = (r_lam - rf_rate) / v_lam

    print(f"\n{'=' * 70}")
    print(f"  三种方法结果汇总")
    print(f"{'=' * 70}")
    print(f"  {'':22} {'无约束MSR':>12} {'有约束MSR':>12} {'风险厌恶λ':>12}")
    print(f"  {'-' * 62}")
    print(f"  {'组合预期收益率':<22} {r_unc:>12.4f} {r_con:>12.4f} {r_lam:>12.4f}")
    print(f"  {'组合波动率':<22} {v_unc:>12.4f} {v_con:>12.4f} {v_lam:>12.4f}")
    print(f"  {'夏普比率':<22} {sr_unc:>12.4f} {sr_con:>12.4f} {sr_lam:>12.4f}")

    print(f"\n  {'资产名称':<20} {'无约束MSR':>12} {'有约束MSR':>12} {'风险厌恶λ':>12}")
    print(f"  {'-' * 58}")
    for i, a in enumerate(assets):
        w1, w2, w3 = w_unc[i], w_con[i], w_lam[i]
        if w1 > 1e-4 or w2 > 1e-4 or w3 > 1e-4:
            print(f"  {a:<20} {w1:>12.2%} {w2:>12.2%} {w3:>12.2%}")
    print(f"{'=' * 70}")


# ==================== 10. 主画图函数 ====================

def plot_three_methods(assets, mu, cov, constraints):
    """
    绘制三张子图：
      左图  - 无约束：全域均匀狄利克雷散点 + 无约束有效前沿
      中图  - 有约束：全域均匀狄利克雷散点（仅保留满足约束的点）+ 有约束有效前沿
      右图  - 风险厌恶：固定 λ 下最优权重 × 20 作为 Dirichlet 参数的集中撒点 + λ 扫描有效前沿
    """

    # -------- 计算三种方法的数据 --------

    print("\n[1/6] 无约束散点云（全域均匀）...")
    rets_unc_sc, vols_unc_sc, sharpes_unc_sc = dirichlet_sampling(mu, cov, constraints=None)

    print("[2/6] 无约束有效前沿 + MSR...")
    fv_unc, fr_unc = calc_frontier_parametric(mu, cov, constraints=None)
    w_unc, r_unc, v_unc = find_tangency_portfolio(mu, cov, constraints=None)
    sr_unc = (r_unc - rf_rate) / v_unc

    print("[3/6] 有约束散点云（全域均匀 → 过滤可行点）...")
    rets_con_sc, vols_con_sc, sharpes_con_sc = dirichlet_sampling(mu, cov, constraints=constraints)

    print("[4/6] 有约束有效前沿 + MSR...")
    fv_con, fr_con = calc_frontier_parametric(mu, cov, constraints=constraints)
    w_con, r_con, v_con = find_tangency_portfolio(mu, cov, constraints=constraints)
    sr_con = (r_con - rf_rate) / v_con

    print("[5/6] 风险厌恶λ扫描有效前沿（有约束）...")
    opt_pts = calc_frontier_lambda(mu, cov, assets, constraints=constraints)
    fv_lam  = np.array([p['vol'] for p in opt_pts])
    fr_lam  = np.array([p['ret'] for p in opt_pts])

    best_lam = max(opt_pts, key=lambda p: p['sharpe'])
    w_lam, r_lam, v_lam = best_lam['weights'], best_lam['ret'], best_lam['vol']
    sr_lam   = best_lam['sharpe']

    # 【修正】固定一个 λ，用其最优权重 × 20 做 Dirichlet 撒点
    # 找到最接近 lambda_for_scatter 的已求解点
    target_pt = min(opt_pts, key=lambda p: abs(p['lambda'] - lambda_for_scatter))
    w_center  = target_pt['weights']
    actual_lambda = target_pt['lambda']
    print(f"  风险厌恶散点：使用 λ={actual_lambda:.2f} 的最优权重 × 20 作为 Dirichlet 参数")

    rets_lam_sc, vols_lam_sc, sharpes_lam_sc = lambda_focused_sampling(
        mu, cov, w_center, num_samples=num_portfolios, concentration=20.0
    )

    # -------- 打印三种方法结果 --------

    print("[6/6] 输出配置结果...")
    print_result("无约束 MSR（最大夏普比率）", assets, mu, w_unc, r_unc, v_unc)
    print_result("有约束 MSR（最大夏普比率）", assets, mu, w_con, r_con, v_con,
                 w_ref=w_unc, label_ref="无约束MSR")
    print_result(f"风险厌恶 λ={best_lam['lambda']:.2f}（有约束）", assets, mu, w_lam, r_lam, v_lam,
                 w_ref=w_con, label_ref="有约束MSR")
    print_three_way_compare(assets, mu,
                             w_unc, r_unc, v_unc,
                             w_con, r_con, v_con,
                             w_lam, r_lam, v_lam)

    # -------- 绘图 --------

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f'三种优化方法有效前沿对比  —  {target_account}',
                 fontsize=15, fontweight='bold')

    vmin = np.percentile(sharpes_unc_sc, 5)
    vmax = np.percentile(sharpes_unc_sc, 95)

    # ==================== 左图：无约束 ====================
    ax1.set_title('① 无约束\n全域均匀狄利克雷散点', fontsize=11)

    sc1 = ax1.scatter(vols_unc_sc, rets_unc_sc, c=sharpes_unc_sc,
                      cmap='Blues', alpha=0.3, s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc1, ax=ax1, label='夏普比率')

    if len(fv_unc) > 1:
        ax1.plot(fv_unc, fr_unc, color='gold', linewidth=2.5, zorder=8, label='有效前沿')

    ax1.scatter(v_unc, r_unc, color='red', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.5,
                label=f'MSR  SR={sr_unc:.3f}')

    ax1.set_xlabel('组合波动率', fontsize=11)
    ax1.set_ylabel('预期收益率', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ==================== 中图：有约束（仅显示可行散点）====================
    ax2.set_title('② 有约束（监管约束）\n仅显示满足约束的散点', fontsize=11)

    sc2 = ax2.scatter(vols_con_sc, rets_con_sc, c=sharpes_con_sc,
                      cmap='Blues', alpha=0.3, s=5, vmin=vmin, vmax=vmax)
    plt.colorbar(sc2, ax=ax2, label='夏普比率')

    if len(fv_con) > 1:
        ax2.plot(fv_con, fr_con, color='tomato', linewidth=2.5, zorder=8, label='有效前沿（有约束）')

    if len(fv_unc) > 1:
        ax2.plot(fv_unc, fr_unc, color='gold', linewidth=1.5, zorder=7,
                 linestyle='--', alpha=0.6, label='无约束前沿（参考）')

    ax2.scatter(v_con, r_con, color='darkred', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.5,
                label=f'MSR  SR={sr_con:.3f}')

    ax2.set_xlabel('组合波动率', fontsize=11)
    ax2.set_ylabel('预期收益率', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ==================== 右图：风险厌恶λ扫描 ====================
    ax3.set_title(f'③ 风险厌恶λ扫描（有约束）\n'
                  f'散点基于 λ={actual_lambda:.1f} 最优权重×20 的 Dirichlet',
                  fontsize=11)

    vmin3 = np.percentile(sharpes_lam_sc, 5)
    vmax3 = np.percentile(sharpes_lam_sc, 95)

    sc3 = ax3.scatter(vols_lam_sc, rets_lam_sc, c=sharpes_lam_sc,
                      cmap='Blues', alpha=0.4, s=5, vmin=vmin3, vmax=vmax3)
    plt.colorbar(sc3, ax=ax3, label='夏普比率')

    if len(fv_lam) > 1:
        ax3.plot(fv_lam, fr_lam, color='gold', linewidth=2.5, zorder=8, label='有效前沿（λ扫描）')

    # 标注关键 λ 值
    for lam_label in [0.1, 1, 5, 20, 100]:
        for p in opt_pts:
            if abs(p['lambda'] - lam_label) / lam_label < 0.15:
                ax3.annotate(
                    f"λ={p['lambda']:.1f}",
                    xy=(p['vol'], p['ret']),
                    xytext=(p['vol'] + 0.003, p['ret']),
                    fontsize=8, color='dimgray',
                    arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.8)
                )
                break

    ax3.scatter(v_lam, r_lam, color='red', marker='*', s=350, zorder=10,
                edgecolors='white', linewidths=0.5,
                label=f'最优λ={best_lam["lambda"]:.2f}  SR={sr_lam:.3f}')

    # 标注散点中心点
    r_center = w_center @ mu
    v_center = np.sqrt(w_center @ cov @ w_center)
    ax3.scatter(v_center, r_center, color='lime', marker='D', s=120, zorder=11,
                edgecolors='black', linewidths=0.8,
                label=f'散点中心 λ={actual_lambda:.1f}')

    ax3.set_xlabel('组合波动率', fontsize=11)
    ax3.set_ylabel('预期收益率', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('frontier_three_methods.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n图已保存为 frontier_three_methods.png")


# ==================== 11. 主程序 ====================

if __name__ == '__main__':
    print(f"\n{'=' * 50}")
    print(f"  资产配置优化  |  账户：{target_account}")
    print(f"{'=' * 50}\n")

    assets, mu, cov = load_data(target_account)
    constraints     = build_constraints(assets)
    print(f"\n约束条件已加载，共 {len(constraints)} 条\n")

    plot_three_methods(assets, mu, cov, constraints)