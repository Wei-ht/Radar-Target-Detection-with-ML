# src/main.py
import numpy as np
import matplotlib.pyplot as plt
from radar_signal import create_dataset
from svm_manual import SVM
from manual_split import manual_train_test_split

import warnings
import random
# 固定所有随机种子以确保可复现性
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings('ignore')

def main():
    X, y, dd_shape = create_dataset()
    print("DD map shape:", dd_shape)

    X_train, X_test, y_train, y_test = manual_train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # 标准化
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    # 训练
    svm = SVM(C=1.0, kernel='rbf', gamma=1e-4)
    svm.fit(X_train, y_train)

    # 评估
    predictions = svm.predict(X_test)
    pred_labels = (predictions > 0).astype(int)
    test_accuracy = np.mean(pred_labels == y_test)
    print(f"Test Accuracy: {test_accuracy:.3f}")

    # 可视化四子图
    pred_1d = predictions.ravel() # 展平为一维数组
    plt.figure(figsize=(12, 8))

    # 子图 1: 距离-多普勒图
    plt.subplot(2, 2, 1)
    plt.imshow(X[0].reshape(dd_shape), cmap='hot', aspect='auto')
    plt.title("Sample Distance-Doppler Map")
    plt.colorbar() # 添加颜色标尺，表示对数幅度值。

    # 子图 2: 预测值分布
    plt.subplot(2, 2, 2)
    plt.hist(pred_1d, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Predicted Values Distribution")

    # 子图 3: 前50个预测值
    plt.subplot(2, 2, 3)
    plt.plot(pred_1d[:50], 'o-', markersize=3, color='tab:blue')
    plt.title("Predictions (First 50 samples)")

    # 子图 4: 预测值 vs 真实标签
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(range(len(pred_1d)), pred_1d, c=y_test, cmap='viridis', alpha=0.7)
    plt.title("Predictions vs True Labels (Test Set)")
    plt.colorbar(scatter, label="True Label (0/1)")

    plt.tight_layout()

    # 保存图像
    import os
    os.makedirs("../results", exist_ok=True)
    save_path = "../results/radar_classification_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    # 可选：显示图像（如在本地运行）
    plt.show()


if __name__ == "__main__":
    main()