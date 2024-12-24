import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
import numpy as np

register_matplotlib_converters()
sns.set_style("darkgrid")

plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)

# 读取CSV文件
data = pd.read_csv("C:/Users/Mr.s/Desktop/data.csv")
period = data["period"]

co2 = period.values

co2 = pd.Series(
    co2, index=pd.date_range("2021-01", periods=len(co2), freq="M"), name="NDVI")

# 根据提供的STL参数进行调整
stl = STL(co2,
          seasonal=13,  # np=46 samples/year (sampling every 8-day)
          robust=True,
          trend=53,  # nt=71, computed and chose the smallest odd number
          seasonal_deg=3,  # ni=1 (number of passes through the inner loop)
          trend_deg=3,
          low_pass=47)  # nl=47, often computed as odd[np]

res = stl.fit()

# 应用滑动平均滤波到季节性分量
window_size = 3
seasonal_filtered = res.seasonal.rolling(window=window_size, center=True).mean()

# 填补时间序列分解后的缺失数据
def fill_missing_values(series):
    series_filled = series.copy()
    for i in range(len(series)):
        if pd.isna(series_filled.iloc[i]):
            if i == 0:
                series_filled.iloc[i] = series_filled.iloc[i + 1]
            elif i == len(series) - 1:
                series_filled.iloc[i] = series_filled.iloc[i - 1]
            else:
                series_filled.iloc[i] = (series_filled.iloc[i - 1] + series_filled.iloc[i + 1]) / 2
    return series_filled

# 填补缺失的趋势和季节性数据
trend_filled = fill_missing_values(res.trend)
seasonal_filled = fill_missing_values(seasonal_filtered)

# 计算预测值（趋势 + 季节性）
predicted_values = trend_filled + seasonal_filled

# 计算MSE
def calculate_mse(true_values, predicted_values):
    return np.mean((true_values - predicted_values) ** 2)

mse_error = calculate_mse(co2, predicted_values)
print(f"Mean Squared Error (MSE): {mse_error:.4f}")

# 提取剩余项
residuals = res.resid

# 绘图
fig1, ax1 = plt.subplots()
ax1.plot(trend_filled, label="Trend")
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.plot(seasonal_filled, label="Seasonal")
ax2.legend()

# 绘制剩余项的正态分布图
fig3, ax3 = plt.subplots()
ax3.hist(residuals, bins=30, density=True)
ax3.set_xlabel("Residual")
ax3.set_ylabel("Density")

plt.show()

# 将趋势性和季节性数据合并到一个DataFrame中
output_df = pd.DataFrame({
    "Date": co2.index,
    "Trend": trend_filled,
    "Seasonal": seasonal_filled
})

# 导出到CSV文件
output_df.to_csv("C:/Users/Mr.s/Desktop/output.csv", index=False)

print("趋势性和季节性分量已导出到output.csv")