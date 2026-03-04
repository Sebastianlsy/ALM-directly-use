"""
================================================================================
SAA 约束影响分析模块（v2 — 含偿付能力挂钩）
================================================================================

更新内容：
- 新增偿付能力充足率与权益上限的挂钩表（来自监管规定）
- 敏感性分析同时展示权益上限和对应的偿付能力档位
- 约束情境设计更贴近真实监管环境

使用方式不变：
    from constraint_analysis_v2 import run_constraint_analysis
    run_constraint_analysis(assets, mu, cov)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

rf_rate = 0.02


# ==================== 0. 偿付能力充足率 ↔ 权益上限 挂钩表 ====================

# 来源：《关于优化保险公司权益类资产配置监管有关事项的通知》
# 综合偿付能力充足率（上季末）→ 权益类资产监管比例上限
SOLVENCY_EQUITY_TABLE = [
    # (充足率下限, 充足率上限, 权益上限)
    (0.00,   1.00,  0.10),   # < 100%  → 10%
    (1.00,   1.50,  0.20),   # 100%-150% → 20%
    (1.50,   2.50,  0.30),   # 150%-250% → 30%
    (2.50,   3.00,  0.35),   # 250%-300% → 35%
    (3.00,   3.50,  0.40),   # 300%-350% → 40%
    (3.50,   9.99,  0.45),   # ≥ 350%  → 45%-50%（取45%）
]

def solvency_to_equity_cap(solvency_ratio):
    """
    根据综合偿付能力充足率查表，返回权益类资产的监管比例上限。

    【参数】
    solvency_ratio: float — 综合偿付能力充足率（如 1.80 表示 180%）

    【返回】
    equity_cap: float — 权益类资产上限（如 0.30 表示 30%）

    【示例】
    solvency_to_equity_cap(1.80)  → 0.30  （180% 在 150%-250% 档，上限 30%）
    solvency_to_equity_cap(2.60)  → 0.35  （260% 在 250%-300% 档，上限 35%）
    """
    for low, high, cap in SOLVENCY_EQUITY_TABLE:
        if low <= solvency_ratio < high:
            return cap
    return 0.45  # 兜底


# 其他大类资产监管上限（来自第一张图）
REGULATORY_CAPS = {
    '不动产类': 0.30,         # 不动产投资上限 30%
    '其他金融资产': 0.25,     # 非标上限 25%
    '境外投资': 0.15,         # 境外投资上限 15%
    '固定收益类': 1.00,       # 无比例上限
}


# ==================== 1. 约束定义工厂 ====================

def build_constraints(scenario, assets, mu,
                      solvency_ratio=1.80,
                      equity_upper=None,
                      re_upper=0.30,
                      nonstandard_upper=0.25,
                      overseas_upper=0.15,
                      liquid_lower=0.05,
                      r_guarantee=0.02,
                      weight_bounds=None):
    """
    根据情境编号返回约束列表。

    【新增参数】
    solvency_ratio: 综合偿付能力充足率，默认 1.80（即 180%）
                    当 equity_upper 未指定时，自动查表确定权益上限
    equity_upper:   手动指定权益上限（优先于 solvency_ratio 查表）
    re_upper:       不动产类上限，默认 0.30
    nonstandard_upper: 其他金融资产（非标）上限，默认 0.25
    overseas_upper: 境外投资上限，默认 0.15

    【情境设计】
    S0: 无约束基准
    S1: + 权益上限（根据偿付能力充足率确定）
    S2: + 不动产、非标、境外上限（全部监管硬约束）
    S3: + 流动性下限
    S4: + 保底收益覆盖
    S5: + 各资产权重区间（全约束）

    ★★★ 需要根据你的实际数据修改资产分类关键词 ★★★
    """

    # 权益上限：优先用手动指定值，否则查偿付能力挂钩表
    if equity_upper is None:
        equity_upper = solvency_to_equity_cap(solvency_ratio)

    # ============================================================
    # ★★★ 修改这里：替换为你 assets 列表中的实际资产名称关键词 ★★★
    # ============================================================
    equity_names     = [a for a in assets if any(kw in a for kw in ['股票', '权益', '基金'])]
    re_names         = [a for a in assets if any(kw in a for kw in ['不动产', '地产', 'REITs'])]
    nonstandard_names = [a for a in assets if any(kw in a for kw in ['非标', '信托', '理财', '资管计划'])]
    overseas_names   = [a for a in assets if any(kw in a for kw in ['境外', '海外', 'QDII'])]
    liquid_names     = [a for a in assets if any(kw in a for kw in ['现金', '货币', '存款', '国债'])]
    # ============================================================

    def name_to_idx(names):
        return [i for i, a in enumerate(assets) if a in names]

    equity_idx      = name_to_idx(equity_names)
    re_idx          = name_to_idx(re_names)
    nonstandard_idx = name_to_idx(nonstandard_names)
    overseas_idx    = name_to_idx(overseas_names)
    liquid_idx      = name_to_idx(liquid_names)

    n = len(assets)
    extra = []
    desc = ""

    # ---- S0: 无约束 ----
    if scenario == 'S0':
        return extra, "无约束基准"

    # ---- S1: 权益上限（偿付能力挂钩） ----
    if scenario >= 'S1':
        if equity_idx:
            extra.append({
                'type': 'ineq',
                'fun': lambda w, idx=equity_idx, ub=equity_upper: ub - sum(w[i] for i in idx)
            })
        solvency_pct = f"{solvency_ratio*100:.0f}%"
        desc = f"权益≤{equity_upper:.0%}（偿付能力{solvency_pct}）"

    if scenario == 'S1':
        return extra, desc

    # ---- S2: + 不动产、非标、境外上限 ----
    if scenario >= 'S2':
        if re_idx:
            extra.append({
                'type': 'ineq',
                'fun': lambda w, idx=re_idx, ub=re_upper: ub - sum(w[i] for i in idx)
            })
        if nonstandard_idx:
            extra.append({
                'type': 'ineq',
                'fun': lambda w, idx=nonstandard_idx, ub=nonstandard_upper: ub - sum(w[i] for i in idx)
            })
        if overseas_idx:
            extra.append({
                'type': 'ineq',
                'fun': lambda w, idx=overseas_idx, ub=overseas_upper: ub - sum(w[i] for i in idx)
            })
        desc = "全部监管硬约束"

    if scenario == 'S2':
        return extra, desc

    # ---- S3: + 流动性下限 ----
    if scenario >= 'S3':
        if liquid_idx:
            extra.append({
                'type': 'ineq',
                'fun': lambda w, idx=liquid_idx, lb=liquid_lower: sum(w[i] for i in idx) - lb
            })
        desc = "+ 流动性下限"

    if scenario == 'S3':
        return extra, desc

    # ---- S4: + 保底收益覆盖 ----
    if scenario >= 'S4':
        extra.append({
            'type': 'ineq',
            'fun': lambda w, m=mu, rg=r_guarantee: w @ m - rg
        })
        desc = "+ 保底收益覆盖"

    if scenario == 'S4':
        return extra, desc

    # ---- S5: + 权重区间约束（全约束） ----
    if scenario >= 'S5':
        if weight_bounds is None:
            for i in range(n):
                extra.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=i: 0.40 - w[idx]
                })
        else:
            for asset_name, (lb_val, ub_val) in weight_bounds.items():
                if asset_name in assets:
                    idx = assets.index(asset_name)
                    if lb_val > 0:
                        extra.append({
                            'type': 'ineq',
                            'fun': lambda w, i=idx, lb=lb_val: w[i] - lb
                        })
                    if ub_val < 1:
                        extra.append({
                            'type': 'ineq',
                            'fun': lambda w, i=idx, ub=ub_val: ub - w[i]
                        })
        desc = "全约束"

    return extra, desc


# ==================== 2. 求解函数 ====================

def calc_frontier_with_constraints(mu, cov, extra_constraints=None, n_points=50):
    """参数化追踪法求有效前沿，支持额外约束。"""
    n = len(mu)
    bounds = [(0, 1)] * n
    target_rets = np.linspace(mu.min(), mu.max() * 0.98, n_points)

    frontier_vols, frontier_rets = [], []

    for target_ret in target_rets:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=target_ret: w @ mu - tr},
        ]
        if extra_constraints:
            constraints.extend(extra_constraints)

        best = None
        for _ in range(8):
            w0 = np.random.dirichlet([1.0] * n)
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

    if len(frontier_vols) > 1:
        mi = np.argmin(frontier_vols)
        frontier_vols = frontier_vols[mi:]
        frontier_rets = frontier_rets[mi:]

    return frontier_vols, frontier_rets


def find_tangency_with_constraints(mu, cov, extra_constraints=None):
    """求带约束的最大夏普比率组合。"""
    n = len(mu)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            return 0
        return -(ret - rf_rate) / vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    bounds = [(0, 1)] * n

    best = None
    for _ in range(30):
        w0 = np.random.dirichlet([1.0] * n)
        res = minimize(
            neg_sharpe, w0, method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return None, None, None, None

    w_opt = best.x
    r_opt = w_opt @ mu
    v_opt = np.sqrt(w_opt @ cov @ w_opt)
    sr_opt = (r_opt - rf_rate) / v_opt if v_opt > 1e-12 else 0
    return w_opt, r_opt, v_opt, sr_opt


# ==================== 3. 全情境分析 ====================

def analyze_all_scenarios(assets, mu, cov, scenarios=None, **kwargs):
    """对所有情境逐一求解。"""
    if scenarios is None:
        scenarios = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']

    results = {}
    for sc in scenarios:
        print(f"\n{'='*50}")
        print(f"  情境 {sc}")
        print(f"{'='*50}")

        extra, desc = build_constraints(sc, assets, mu, **kwargs)
        print(f"  描述: {desc}")
        print(f"  额外约束数: {len(extra)}")

        fv, fr = calc_frontier_with_constraints(mu, cov, extra_constraints=extra)
        print(f"  前沿点数: {len(fv)}")

        w_opt, r_opt, v_opt, sr_opt = find_tangency_with_constraints(mu, cov, extra_constraints=extra)

        if w_opt is not None:
            hhi = np.sum(w_opt ** 2)
            max_w = np.max(w_opt)
            print(f"  夏普比率: {sr_opt:.4f}")
            print(f"  HHI: {hhi:.4f}")
        else:
            hhi, max_w = None, None
            print(f"  求解失败")

        results[sc] = {
            'desc': desc,
            'frontier_vols': fv, 'frontier_rets': fr,
            'w_opt': w_opt, 'r_opt': r_opt, 'v_opt': v_opt,
            'sr_opt': sr_opt, 'hhi': hhi, 'max_weight': max_w,
        }
    return results


# ==================== 4. 偿付能力挂钩专项分析 ====================

def solvency_scenario_analysis(assets, mu, cov, base_scenario='S5', **kwargs):
    """
    偿付能力充足率情景分析：
    按监管挂钩表的 6 个档位，分别计算 SAA 最优结果。

    这是本报告的特色分析——把权益上限和偿付能力直接挂钩，
    回答"偿付能力每提高一个档位，SAA效率能提升多少"。
    """
    # 取每个档位的典型值
    solvency_scenarios = [
        (0.80, "< 100%（困难期）"),
        (1.20, "100%-150%（关注期）"),
        (1.80, "150%-250%（正常）"),
        (2.70, "250%-300%（充裕）"),
        (3.20, "300%-350%（优秀）"),
        (3.80, "\u2265 350%（极充裕）"),
    ]

    results = {}
    print("\n" + "█" * 60)
    print("  偿付能力充足率情景分析")
    print("█" * 60)

    for sr_val, label in solvency_scenarios:
        eq_cap = solvency_to_equity_cap(sr_val)
        kw = kwargs.copy()
        kw['solvency_ratio'] = sr_val
        kw['equity_upper'] = eq_cap

        extra, desc = build_constraints(base_scenario, assets, mu, **kw)
        w_opt, r_opt, v_opt, sr_opt = find_tangency_with_constraints(mu, cov, extra_constraints=extra)
        fv, fr = calc_frontier_with_constraints(mu, cov, extra_constraints=extra, n_points=40)

        key = f"充足率{sr_val*100:.0f}%"
        results[key] = {
            'solvency_ratio': sr_val,
            'equity_cap': eq_cap,
            'label': label,
            'frontier_vols': fv, 'frontier_rets': fr,
            'w_opt': w_opt, 'r_opt': r_opt, 'v_opt': v_opt,
            'sr_opt': sr_opt,
            'hhi': np.sum(w_opt ** 2) if w_opt is not None else None,
        }
        sr_str = f"{sr_opt:.4f}" if sr_opt is not None else "失败"
        print(f"  {label:<22} 权益上限={eq_cap:.0%}  SR={sr_str}")

    # ---- 可视化 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('偿付能力充足率对SAA效率的影响', fontsize=14, fontweight='bold')

    # 左图：多前沿叠加
    colors = ['#D32F2F', '#F57C00', '#FBC02D', '#388E3C', '#1976D2', '#7B1FA2']
    for i, (key, data) in enumerate(results.items()):
        fv, fr = data['frontier_vols'], data['frontier_rets']
        if len(fv) < 2:
            continue
        ax1.plot(fv, fr, color=colors[i % len(colors)], linewidth=2,
                 label=f"权益≤{data['equity_cap']:.0%} ({data['label'][:6]})")
        if data['w_opt'] is not None:
            ax1.scatter(data['v_opt'], data['r_opt'], color=colors[i % len(colors)],
                        marker='*', s=200, zorder=10, edgecolors='white', linewidths=0.5)

    ax1.set_xlabel('组合波动率 σ', fontsize=11)
    ax1.set_ylabel('预期收益率', fontsize=11)
    ax1.set_title('有效前沿随偿付能力变化', fontsize=11)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 右图：偿付能力充足率 vs 夏普比率（阶梯图）
    sol_ratios = [d['solvency_ratio'] * 100 for d in results.values()]
    eq_caps    = [d['equity_cap'] * 100 for d in results.values()]
    srs        = [d['sr_opt'] if d['sr_opt'] is not None else 0 for d in results.values()]

    ax2.bar(range(len(results)), srs, color=colors[:len(results)], edgecolor='gray', linewidth=0.5)

    # 双标签：上面标权益上限，下面标充足率档位
    ax2.set_xticks(range(len(results)))
    labels = [f"权益≤{ec:.0f}%\n充足率{sr:.0f}%" for ec, sr in zip(eq_caps, sol_ratios)]
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('夏普比率', fontsize=11)
    ax2.set_title('偿付能力档位 vs SAA效率', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 标注SR值
    for i, sr in enumerate(srs):
        ax2.text(i, sr + 0.003, f"{sr:.4f}", ha='center', fontsize=9, fontweight='bold')

    # 标注档位间的SR增量
    for i in range(1, len(srs)):
        delta = srs[i] - srs[i-1]
        if abs(delta) > 1e-6:
            mid_y = (srs[i] + srs[i-1]) / 2
            ax2.annotate(
                f"+{delta:.4f}",
                xy=(i - 0.5, mid_y),
                fontsize=7, color='red', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='gray', alpha=0.8)
            )

    plt.tight_layout()
    plt.savefig('solvency_impact.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 solvency_impact.png")

    # ---- 汇总表 ----
    print("\n" + "=" * 85)
    print("  偿付能力情景汇总表")
    print("=" * 85)
    print(f"{'充足率':<10} {'档位描述':<22} {'权益上限':>8} {'收益率':>8} {'波动率':>8} {'SR':>8} {'HHI':>8}")
    print("-" * 85)
    for key, d in results.items():
        if d['sr_opt'] is not None:
            print(f"{d['solvency_ratio']*100:>6.0f}%   {d['label']:<22} "
                  f"{d['equity_cap']:>8.0%} {d['r_opt']:>8.4f} {d['v_opt']:>8.4f} "
                  f"{d['sr_opt']:>8.4f} {d['hhi']:>8.4f}")
    print("=" * 85)

    return results


# ==================== 5. 通用可视化 ====================

def plot_frontier_comparison(results, title_suffix=""):
    """多情境有效前沿叠加图。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title(f'SAA有效前沿：约束逐层叠加{title_suffix}', fontsize=13, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#607D8B']
    linestyles = ['-', '--', '-.', ':', '-', '--']

    for i, (sc, data) in enumerate(results.items()):
        fv, fr = data['frontier_vols'], data['frontier_rets']
        if len(fv) < 2:
            continue
        ax.plot(fv, fr, color=colors[i % len(colors)], linewidth=2,
                linestyle=linestyles[i % len(linestyles)],
                label=f"{sc}: {data['desc']}", zorder=5+i)
        if data['w_opt'] is not None:
            ax.scatter(data['v_opt'], data['r_opt'], color=colors[i % len(colors)],
                       marker='*', s=200, zorder=10+i, edgecolors='white', linewidths=0.5)

    ax.set_xlabel('组合波动率 σ', fontsize=11)
    ax.set_ylabel('预期收益率', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('frontier_constraint_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 frontier_constraint_comparison.png")


def plot_weight_evolution(assets, results):
    """权重变化堆积柱状图。"""
    scenarios = [sc for sc in results if results[sc]['w_opt'] is not None]
    if not scenarios:
        return

    n_scenarios = len(scenarios)
    weight_matrix = np.zeros((n_scenarios, len(assets)))
    for i, sc in enumerate(scenarios):
        weight_matrix[i] = results[sc]['w_opt']

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_title('最优组合权重变化：约束逐层叠加', fontsize=13, fontweight='bold')

    active = np.any(weight_matrix > 0.01, axis=0)
    active_assets = [a for a, f in zip(assets, active) if f]
    active_weights = weight_matrix[:, active]

    x = np.arange(n_scenarios)
    bottom = np.zeros(n_scenarios)
    colors = plt.cm.Set3(np.linspace(0, 1, len(active_assets)))

    for j, (name, color) in enumerate(zip(active_assets, colors)):
        ax.bar(x, active_weights[:, j], bottom=bottom, label=name,
               color=color, edgecolor='white', linewidth=0.5)
        bottom += active_weights[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sc}\n{results[sc]['desc'][:12]}" for sc in scenarios], fontsize=8)
    ax.set_ylabel('权重', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('weight_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 weight_evolution.png")


def plot_sr_waterfall(results):
    """夏普比率递减瀑布图。"""
    scenarios = [sc for sc in results if results[sc]['sr_opt'] is not None]
    if len(scenarios) < 2:
        return

    srs = [results[sc]['sr_opt'] for sc in scenarios]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title('夏普比率递减瀑布图', fontsize=13, fontweight='bold')

    colors = ['#2196F3' if i == 0 else ('#E91E63' if i == len(scenarios)-1 else '#B0BEC5')
              for i in range(len(scenarios))]
    bars = ax.bar(range(len(scenarios)), srs, color=colors, edgecolor='gray', linewidth=0.5)

    for i, (bar, sr) in enumerate(zip(bars, srs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{sr:.4f}", ha='center', fontsize=9)

    for i in range(1, len(scenarios)):
        delta = srs[i] - srs[i-1]
        if abs(delta) > 1e-6:
            ax.annotate(f"Δ={delta:+.4f}", xy=(i, srs[i]),
                        xytext=(i, srs[i] + 0.015), fontsize=8, ha='center',
                        color='red' if delta < 0 else 'green',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([f"{sc}\n{results[sc]['desc'][:10]}" for sc in scenarios], fontsize=8)
    ax.set_ylabel('夏普比率', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('sr_waterfall.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图已保存为 sr_waterfall.png")


# ==================== 6. 结果汇总 ====================

def print_summary_table(assets, results):
    """打印约束代价汇总表。"""
    scenarios = list(results.keys())
    sr_base = results.get('S0', {}).get('sr_opt', None)

    print("\n" + "=" * 90)
    print("  SAA 约束影响分析汇总表")
    print("=" * 90)
    print(f"{'情境':<6} {'描述':<25} {'收益率':>8} {'波动率':>8} {'SR':>8} {'ΔSR':>8} {'HHI':>8}")
    print("-" * 90)

    for sc in scenarios:
        d = results[sc]
        if d['sr_opt'] is None:
            print(f"{sc:<6} {d['desc']:<25} {'失败':>8}")
            continue
        delta_sr = d['sr_opt'] - sr_base if sr_base is not None else 0
        print(f"{sc:<6} {d['desc']:<25} "
              f"{d['r_opt']:>8.4f} {d['v_opt']:>8.4f} "
              f"{d['sr_opt']:>8.4f} {delta_sr:>+8.4f} "
              f"{d['hhi']:>8.4f}")

    print("=" * 90)

    # 边际代价排名
    if sr_base is not None and len(scenarios) > 1:
        print("\n  约束边际代价排名：")
        margins = []
        for i in range(1, len(scenarios)):
            sr_prev = results[scenarios[i-1]].get('sr_opt')
            sr_curr = results[scenarios[i]].get('sr_opt')
            if sr_prev is not None and sr_curr is not None:
                margins.append((scenarios[i], results[scenarios[i]]['desc'], sr_curr - sr_prev))
        margins.sort(key=lambda x: x[2])
        for rank, (sc, desc, delta) in enumerate(margins, 1):
            bar = "█" * max(1, int(abs(delta) * 500))
            print(f"  {rank}. {sc} ({desc}): ΔSR = {delta:+.4f}  {bar}")

    # S0 vs 全约束 权重对比
    last_sc = scenarios[-1]
    if 'S0' in results and last_sc in results:
        w0 = results['S0']['w_opt']
        wN = results[last_sc]['w_opt']
        if w0 is not None and wN is not None:
            print(f"\n  权重对比：S0 vs {last_sc}")
            print(f"  {'资产':<18} {'S0':>10} {last_sc:>10} {'变化':>10}")
            print(f"  {'-'*50}")
            for i, name in enumerate(assets):
                if w0[i] > 1e-4 or wN[i] > 1e-4:
                    print(f"  {name:<18} {w0[i]:>10.2%} {wN[i]:>10.2%} {w0[i]-wN[i]:>+10.2%}")


# ==================== 7. 一键运行入口 ====================

def run_constraint_analysis(assets, mu, cov, solvency_ratio=1.80, **kwargs):
    """
    一键运行完整分析，包括：
    1. 逐层约束分析（S0-S5）
    2. 偿付能力充足率情景分析（6个档位）
    3. 全部可视化和汇总表

    【参数】
    assets, mu, cov:  你原始代码的 load_data() 返回值
    solvency_ratio:   当前偿付能力充足率，默认 1.80（180%）
    **kwargs:         其他参数（liquid_lower, r_guarantee 等）
    """
    print("\n" + "█" * 60)
    print(f"  SAA 约束影响分析")
    print(f"  偿付能力充足率: {solvency_ratio*100:.0f}%")
    print(f"  对应权益上限: {solvency_to_equity_cap(solvency_ratio):.0%}")
    print("█" * 60)

    # ---- Part 1: 逐层约束分析 ----
    print("\n\n" + "=" * 60)
    print("  Part 1: 逐层约束分析")
    print("=" * 60)

    results = analyze_all_scenarios(
        assets, mu, cov,
        scenarios=['S0', 'S1', 'S2', 'S3', 'S4', 'S5'],
        solvency_ratio=solvency_ratio,
        **kwargs
    )

    print_summary_table(assets, results)
    plot_frontier_comparison(results)
    plot_weight_evolution(assets, results)
    plot_sr_waterfall(results)

    # ---- Part 2: 偿付能力情景分析 ----
    print("\n\n" + "=" * 60)
    print("  Part 2: 偿付能力充足率情景分析")
    print("=" * 60)

    solvency_results = solvency_scenario_analysis(
        assets, mu, cov,
        base_scenario='S5',
        **kwargs
    )

    print("\n" + "█" * 60)
    print("  分析完成！输出文件：")
    print("  - frontier_constraint_comparison.png  多前沿叠加图")
    print("  - weight_evolution.png                权重变化图")
    print("  - sr_waterfall.png                    夏普比率瀑布图")
    print("  - solvency_impact.png                 偿付能力影响图")
    print("█" * 60)

    return results, solvency_results
