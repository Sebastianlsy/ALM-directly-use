import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# ==================== 1. 基础配置 ====================
FILE_MAIN = "中间项.xlsx"
FILE_RISK = "相关性矩阵.xlsx"
TARGET_ACCOUNT = '传统账户'

# 解决绘图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 2. 数据读取与自动对齐 ====================
def load_data():
    # 读取收益率 (从 中间项.xlsx 的 '量化指标' Sheet)
    df_quant = pd.read_excel(FILE_MAIN, sheet_name='量化指标')
    mask = (df_quant['保险账户分类/InsuranceAccountType'] == TARGET_ACCOUNT) & \
           (~df_quant['资产配置分类三级/SAAAssetTypeLevel3'].str.contains('负债', na=False))
    df_q = df_quant[mask].copy()
    mu_series = df_q.groupby('资产配置分类三级/SAAAssetTypeLevel3')['预期投资收益率/ExpectedReturn'].mean()

    # 读取相关性 (从 相关性矩阵.xlsx 的 '相关性' Sheet)
    df_corr_raw = pd.read_excel(FILE_RISK, sheet_name='相关性', index_col=0)
    if '负债' in df_corr_raw.index:
        df_corr_raw = df_corr_raw.drop(index='负债', columns='负债', errors='ignore')
    corr_matrix = df_corr_raw.combine_first(df_corr_raw.T).fillna(1.0)

    # 读取波动率 (从 相关性矩阵.xlsx 的 '自定义指标' Sheet)
    df_vol_raw = pd.read_excel(FILE_RISK, sheet_name='自定义指标').set_index('投资组合名称/PortfolioName')

    # 三表资产对齐
    assets = mu_series.index.intersection(corr_matrix.index).intersection(df_vol_raw.index)

    mu = mu_series.loc[assets].values
    vols = df_vol_raw.loc[assets, '波动率/Volatility'].values
    corr = corr_matrix.loc[assets, assets].values

    # 构建协方差矩阵
    cov = np.diag(vols) @ corr @ np.diag(vols)

    return list(assets), mu, cov


# ==================== 3. 核心计算函数 ====================
def port_stats(w, mu, cov):
    p_ret = np.dot(w, mu)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    return p_vol, p_ret


def min_vol_for_ret(w, mu, cov):
    return port_stats(w, mu, cov)[0]


def neg_sharpe(w, mu, cov, rf=0):
    vol, ret = port_stats(w, mu, cov)
    return -(ret - rf) / vol


# ==================== 4. 主程序：模拟与优化 ====================
assets, mu, cov = load_data()
n = len(assets)

# A. 狄利克雷撒点模拟
weights_sim = np.random.dirichlet([1.0] * n, 10000)
rets_sim = np.dot(weights_sim, mu)
vols_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov, weights_sim))

# B. 求解有效前沿 (红线)
target_rets = np.linspace(rets_sim.min(), rets_sim.max(), 30)
frontier_vols = []
for tr in target_rets:
    cons = ({'type': 'eq', 'fun': lambda x: np.dot(x, mu) - tr},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(min_vol_for_ret, n * [1. / n], args=(mu, cov),
                       method='SLSQP', bounds=tuple((0, 1) for _ in range(n)), constraints=cons)
    frontier_vols.append(res.fun if res.success else np.nan)

# C. 求解最优夏普组合 (图中黄色星星)
res_sharpe = sco.minimize(neg_sharpe, n * [1. / n], args=(mu, cov),
                          method='SLSQP', bounds=tuple((0, 1) for _ in range(n)),
                          constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))
opt_w = res_sharpe.x
opt_vol, opt_ret = port_stats(opt_w, mu, cov)

# ==================== 5. 绘图与结果打印 ====================
plt.figure(figsize=(12, 7))
plt.scatter(vols_sim, rets_sim, c=rets_sim / vols_sim, cmap='viridis', alpha=0.3, s=10)
plt.plot(frontier_vols, target_rets, 'r--', linewidth=3, label='有效前沿')
plt.scatter(opt_vol, opt_ret, color='gold', marker='*', s=300, label='最大夏普组合', edgecolors='black')

plt.title(f"资产配置模型全景图 - {TARGET_ACCOUNT}", fontsize=15)
plt.xlabel("波动率 (风险)")
plt.ylabel("预期收益率")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 打印最优资产配置表
print("\n" + "=" * 40)
print(f"--- {TARGET_ACCOUNT} 最优资产配置方案 (Max Sharpe) ---")
print(f"{'资产名称':<20} | {'权重比例':>8}")
print("-" * 40)
for a, w in zip(assets, opt_w):
    if w > 0.0001:  # 只打印权重不为 0 的资产
        print(f"{a:<20} | {w * 100:>8.2f}%")
print("-" * 40)
print(f"预期收益率: {opt_ret * 100:.2f}%")
print(f"预期波动率: {opt_vol * 100:.2f}%")
print(f"夏普比率: {opt_ret / opt_vol:.2f}")
print("=" * 40)