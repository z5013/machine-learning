"""
@title robust scaler normalization
@data: 2025/05/18
@author: weidongfeng
"""
import numpy as np
def robust_scaler(data):
    """
    对输入的二维numpy数组应用Robust Scaler。

    参数:
        data (numpy.ndarray): 需要进行缩放的数据，形状为(n_samples, n_features)。

    返回:
        numpy.ndarray: 缩放后的数据。
    """
    median = np.median(data,axis=0)
    # 计算第一四分位数和第三四分位数
    Q1 = np.percentile(data,25,axis=0)
    Q3 = np.percentile(data,75,axis=0)
    # 计算IQR
    IRQ = Q3 - Q1
    scale_data = (data - median) / IRQ
    return scale_data

# 示例数据
data = np.array([[1, 2],
                 [3, 4],
                 [5, 6],
                 [7, 8],
                 [9, 10],
                 [100, 200]])

# 应用自定义的Robust Scaler
scaled_data = robust_scaler(data)

print("Original Data:\n", data)
print("\nScaled Data using Robust Scaler:\n", scaled_data)

from sklearn.preprocessing import RobustScaler
import numpy as np

# 假设我们有一个特征向量
data = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]])

# 初始化RobustScaler
scaler = RobustScaler()

# 对数据进行fit和transform操作
scaled_data = scaler.fit_transform(data)

print("Scaled Data:\n", scaled_data)
