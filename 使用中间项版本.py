import pandas as pd
import numpy as np
# from pypfopt import EfficientFrontier

# ------------------------------------ 参数设置 --------------------------
files_name = "中间项.xlsx"
target_account = '传统账户'
# ---------------------------------------------------------------------


# --------------------------- 数据清洗 ----------------------------------
# 1. 读取文件
df = pd.read_excel(files_name,sheet_name = 0, header = 0)
print("实际列名如下：")
print(df.columns.tolist())

# 2. 保留所需资产列表
## 资产分裂
column_wanted = ['保险账户分类/InsuranceAccountType',
                 # '资产配置分类一级/SAAAssetTypeLevel1',
                 # '资产配置分类二级/SAAAssetTypeLevel2',
                 '资产配置分类三级/SAAAssetTypeLevel3',
                 '全价市场价值/DirtyMarketValue',
                 '预期投资收益率/ExpectedReturn',
                 # '波动率/Volatility',
                 '在险价值99.5%/VaR99.5%',
                 '非流动性溢价/IlliquidityPremium',
                 '修正久期/ModifiedDuration',
                 '凸性/Convexity',
                 '信用利差/Spread',
                 '到期期限/Maturity']

# 3. 重命名列名
new_df = df[column_wanted].copy()
new_df = new_df.rename(columns={
    '保险账户分类/InsuranceAccountType':'Account type',
    # '资产配置分类一级/SAAAssetTypeLevel1':'Asset type Level 1',
    # '资产配置分类二级/SAAAssetTypeLevel2':'Asset type Level 2',
    '资产配置分类三级/SAAAssetTypeLevel3':'Asset type Level 3',
    '全价市场价值/DirtyMarketValue':'Market Value',
    '预期投资收益率/ExpectedReturn':'Expected Return',
    # '波动率/Volatility':'Volatility',
    '在险价值99.5%/VaR99.5%':'VaR99.5%',
    '非流动性溢价/IlliquidityPremium':'Illiquidity Premium',
    '修正久期/ModifiedDuration':'Modified Duration',
    '凸性/Convexity':'Convexity',
    '信用利差/Spread':'Spread',
    '到期期限/Maturity':'Maturity'})

# 筛选目标账户：eg. 传统账户
new_df = new_df[new_df["Account type"] == target_account].copy()
# 清理无效数据
new_df = new_df.dropna(subset=["Market Value"])     #市值不能为0
new_df = new_df[new_df["Market Value"] > 0]     # 市值得为正
print(new_df)

# 导出数据
new_df.to_excel('traditional_account_data.xlsx', index=False)

print(f"成功提取{target_account}数据, 共计{len(new_df)}行")
# --------------------------------------------------------------------------


#------------------------------ 计算资产协方差矩阵 ---------------------
# 读取两个文件
trad_df = pd.read_excel('traditional_account_data.xlsx')
corr_df = pd.read_excel('相关性矩阵.xlsx', sheet_name = 4)
vol_df = pd.read_excel("相关性矩阵.xlsx", sheet_name = 6)
# 只保留前两列
vol_df = vol_df.iloc[:, :2]

# 映射两个表格中不一样的内容
name_map = {"高等级信用债_3Y":"信用债_3Y",
            "高等级信用债_5Y":"信用债_5Y",
            "高等级信用债_10Y":"信用债_10Y"}

# 处理预期收益率
trad_df_clean = trad_df[['Asset type Level 3','Expected Return' ]].copy()
# 将表格中不一样的名字进行替换
trad_df_clean['asset_std'] = trad_df_clean['Asset type Level 3'].map(lambda x: name_map.get(x,x))
mu_series = trad_df_clean.set_index("asset_std")["Expected Return"]
print(trad_df_clean.head())

# 处理波动率
vol_df.columns = ['asset_std','volatility']
vol_series = vol_df.set_index("asset_std")["volatility"]

common_asset = mu_series.index.intersection(vol_series.index).intersection(corr_df.index)
print(f'三者共有资产{list(common_asset)}')

# 按照common_assets 的顺序提取数据
asset = list(common_asset)
mu = mu_series.loc[asset].values
sigma = vol_series.loc[asset].values
corr = corr_df.loc[asset, asset].values

# 构建协方差矩阵
D = np.diag(sigma)
cov_matrix = D @ corr @ D

cov_df = pd.DataFrame(cov_matrix, index = asset, columns = asset)
print("协方差矩阵构建完毕")
print(cov_df.head())

