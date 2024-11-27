from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 假设你有一个numpy数组，包含你的标签数据
labels = np.array([0, 1, 2, 0, 1, 2])

# 创建OneHotEncoder实例
encoder = OneHotEncoder(sparse=False)  # sparse=False意味着输出一个numpy数组，而不是稀疏矩阵

# 将标签数据转换为one-hot编码
one_hot_labels = encoder.fit_transform(labels.reshape(-1, 1))  # reshape是必要的，因为OneHotEncoder期望2D输入

print(one_hot_labels)