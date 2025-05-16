"""
# 线性归一化（Min-Max Scaling）简介

线性归一化是一种常用的数据预处理方法，将数据缩放到指定范围（如 [0, 1] 或 [-1, 1]），便于不同量纲或尺度的数据比较和建模。

## 应用场景

- **机器学习**：提升KNN、SVM、神经网络等模型的训练效率与性能。
- **图像处理**：增强对比度、优化边缘检测效果。
- **时间序列分析**：消除单位差异，便于多序列对比。
- **数据分析与可视化**：更直观展示多维数据分布。

## 局限性

- **对异常值敏感**：最大最小值易受极端值影响，压缩正常值范围。
- **不改变分布形状**：无法修正偏态分布等问题。
- **适用于有界数据**：不适合无明确边界或剧烈变化的数据。
- **信息丢失风险**：可能影响需保留绝对数值的应用（如金融分析）。

## 总结

线性归一化简单高效，但需结合数据特性判断其适用性。必要时可搭配标准化、稳健缩放等方法，以提升整体数据质量。选择合适的预处理方式是建模成功的关键之一。
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_random_data(size=100, low=1, high=1000):
    """生成随机整数数组作为示例数据"""
    return np.random.randint(low, high, size=size)


def min_max_normalize(data_array):
    """
    对输入数组进行线性归一化处理，返回归一化后的数组。
    参数:
        data_array (np.array or list): 原始数据
    返回:
        normalized_data (np.array): 归一化后在 [0, 1] 范围内的数据
    """
    data_min = np.min(data_array)
    data_max = np.max(data_array)
    epsilon = 1e-8
    return (data_array - data_min) / (data_max - data_min + epsilon)


def plot_data_distributions(raw_data, normalized_data):
    """绘制原始数据和归一化后数据的直方图"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].hist(raw_data, bins=50, color='skyblue', edgecolor='black')
    axs[0].set_title('Original Data Distribution')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(normalized_data, bins=50, color='salmon', edgecolor='black')
    axs[1].set_title('Normalized Data Distribution ([0, 1])')
    axs[1].set_xlabel('Normalized Value')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例数据生成
    raw_data = generate_random_data()
    print(f"原始数据的最小值: {np.min(raw_data)}")
    print(f"原始数据的最大值: {np.max(raw_data)}")

    # 数据归一化
    normalized_data = min_max_normalize(raw_data)
    print(f"归一化后的最小值: {np.min(normalized_data)}")
    print(f"归一化后的最大值: {np.max(normalized_data)}")

    # 绘制数据分布
    plot_data_distributions(raw_data, normalized_data)