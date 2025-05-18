"""
线性归一化（Min-Max Scaling）处理二维数据

支持：
- 按列归一化（适用于特征归一化）
- 按行归一化（适用于样本归一化）

公式：
对于输入值 x，归一化到 [0, 1] 区间的方式为：
           x - min(x)
x' = ---------------------
     max(x) - min(x) + ε

其中 ε 是一个极小量（如 1e-8），用于防止除以零。
"""

import numpy as np
import matplotlib.pyplot as plt


def min_max_normalize_2d_cols(data_2d):
    """
    对输入二维数组按**列**进行线性归一化，返回归一化后的二维数组。

    公式：(X[:,j] - min(X[:,j])) / (max(X[:,j]) - min(X[:,j]) + ε)

    参数:
        data_2d (np.ndarray or list): 原始二维数据，形状为 (n_samples, n_features)
    返回:
        normalized_data_2d (np.ndarray): 每列独立归一化到 [0, 1] 的二维数组
    """
    data_min = np.min(data_2d, axis=0)  # 获取每列的最小值
    data_max = np.max(data_2d, axis=0)  # 获取每列的最大值
    epsilon = 1e-8  # 防止除以零的小偏移量
    normalized_data_2d = (data_2d - data_min) / (data_max - data_min + epsilon)
    return normalized_data_2d


def min_max_normalize_2d_rows(data_2d):
    """
    对输入二维数组按**行**进行线性归一化，返回归一化后的二维数组。

    公式：(X[i,:] - min(X[i,:])) / (max(X[i,:]) - min(X[i,:]) + ε)

    参数:
        data_2d (np.ndarray or list): 原始二维数据，形状为 (n_samples, n_features)
    返回:
        normalized_data_2d (np.ndarray): 每行独立归一化到 [0, 1] 的二维数组
    """
    data_min = np.min(data_2d, axis=1).reshape(-1, 1)  # 每行最小值并扩展维度
    data_max = np.max(data_2d, axis=1).reshape(-1, 1)  # 每行最大值并扩展维度
    epsilon = 1e-8
    normalized_data_2d = (data_2d - data_min) / (data_max - data_min + epsilon)
    return normalized_data_2d


def plot_feature_distributions(original, normalized, feature_names=None):
    """
    绘制原始和归一化后各特征的分布直方图，便于对比分析。

    参数:
        original (np.ndarray): 归一化前的数据
        normalized (np.ndarray): 归一化后的数据
        feature_names (list): 特征名称列表（可选）
    """
    n_features = original.shape[1]

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]

    fig, axes = plt.subplots(n_features, 2, figsize=(12, 3 * n_features))

    for i in range(n_features):
        # 原始数据直方图
        axes[i, 0].hist(original[:, i], bins=50, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f'Original: {feature_names[i]}')
        axes[i, 0].set_xlabel('Value')
        axes[i, 0].set_ylabel('Frequency')

        # 归一化后数据直方图
        axes[i, 1].hist(normalized[:, i], bins=50, color='salmon', edgecolor='black')
        axes[i, 1].set_title(f'Normalized: {feature_names[i]}')
        axes[i, 1].set_xlabel('Normalized Value')
        axes[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 生成示例二维数据：100个样本，3个特征（例如身高、体重、年龄）
    raw_data = np.random.randint(1, 1000, size=(100, 3))
    print("原始数据范围（每列）：")
    print(f"最小值: {np.min(raw_data, axis=0)}")
    print(f"最大值: {np.max(raw_data, axis=0)}")

    # 按列进行 Min-Max 归一化
    normalized_data = min_max_normalize_2d_cols(raw_data)

    print("\n归一化后数据范围（每列）：")
    print(f"最小值: {np.min(normalized_data, axis=0)}")
    print(f"最大值: {np.max(normalized_data, axis=0)}")

    # 可视化每个特征在归一化前后的分布变化
    plot_feature_distributions(
        raw_data,
        normalized_data,
        feature_names=["Height", "Weight", "Age"]
    )