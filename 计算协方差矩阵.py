import pandas as pd
import numpy as np

# 1. 读取数据
# 假设从“相关性矩阵.xlsx”中读取这两个 Sheet
df_corr = pd.read_excel("相关性矩阵.xlsx", sheet_name="相关性")
df_vol = pd.read_excel("相关性矩阵.xlsx", sheet_name="自定义指标")

# 2. 清洗相关性矩阵
# 设置索引并确保它是一个对称方阵（处理掉只有下三角或含有空值的情况）
df_corr = df_corr.set_index(df_corr.columns[0])
# 填充缺失值（如果是下三角矩阵，需要填充成全矩阵）
corr_matrix = df_corr.combine_first(df_corr.T).fillna(1.0)

# 3. 清洗波动率数据
# 假设第一列是资产名，第二列是波动率
vol_series = df_vol.set_index(df_vol.columns[0])['波动率/Volatility']

# 4. 取两者的交集资产，确保顺序完全一致
common_assets = corr_matrix.index.intersection(vol_series.index)
R = corr_matrix.loc[common_assets, common_assets].values
vols = vol_series.loc[common_assets].values

# 5. 计算协方差矩阵
D = np.diag(vols)
cov_matrix = D @ R @ D

# 6. 转换为 DataFrame 并导出
df_cov = pd.DataFrame(cov_matrix, index=common_assets, columns=common_assets)
df_cov.to_excel("协方差矩阵.xlsx")

print("协方差矩阵已计算完成并导出。")