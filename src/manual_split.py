# src/manual_split.py
import numpy as np

def manual_train_test_split(X, y, test_size=0.4, random_state=42):
    """
    NumPy 实现的分层抽样数据集划分。

    参数:
        X: 特征矩阵，shape (n_samples, n_features)
        y: 标签向量，shape (n_samples,)
        test_size: 测试集比例
        random_state: 随机种子

    返回:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)

    # 获取唯一标签及其数量
    unique_labels, counts = np.unique(y, return_counts=True)
    train_idx, test_idx = [], []

    for label, count in zip(unique_labels, counts):
        # 找出当前类别的所有样本索引
        label_indices = np.where(y == label)[0]
        # 打乱顺序
        np.random.shuffle(label_indices)
        # 计算测试集大小（向下取整）
        n_test = int(count * test_size)
        # 划分
        test_idx.extend(label_indices[:n_test])
        train_idx.extend(label_indices[n_test:])

    # 转为数组并打乱整体顺序（可选，但更规范）
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]