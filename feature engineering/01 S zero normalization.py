import numpy as np

data = np.array([2, 4, 6, 8, 10])

# 方法一：手动实现
mu = np.mean(data)
sigma = np.std(data)
z_scores = (data - mu) / sigma

print("Z-Scores:", z_scores)

# 方法二：使用 sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_2d = data.reshape(-1, 1)  # 转成二维数组
z_scores_sklearn = scaler.fit_transform(data_2d)

print("Z-Scores (sklearn):", z_scores_sklearn.flatten())