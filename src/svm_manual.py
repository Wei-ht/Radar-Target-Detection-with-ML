# src/svm_manual.py
import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.support_vectors = None
        self.alphas = None
        self.b = 0.0
        self.support_vector_labels = None

    def _kernel(self, X, Y):
        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (X.shape[1] * np.var(X))
            else:
                gamma = self.gamma
            return np.exp(-gamma * np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=2))
        else:
            raise ValueError("Unsupported kernel")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 0, -1, 1) # 标签转换
        alphas = np.zeros(n_samples)
        b = 0.0

        # >>>>> 预计算整个核矩阵 <<<<<
        K_full = self._kernel(X, X)  # shape (n, n),预计算核矩阵

        for iteration in range(100):
            changed = False
            for i in range(n_samples):
                f_xi = np.sum(alphas * y * K_full[:, i]) + b
                E_i = f_xi - y[i]
                if abs(E_i) > 1e-3:
                    j = np.random.choice([k for k in range(n_samples) if k != i])
                    f_xj = np.sum(alphas * y * K_full[:, j]) + b
                    E_j = f_xj - y[j]

                    #求aj的最小L、最大H
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - self.C)
                        H = min(self.C, alphas[j] + alphas[i])
                    if L == H:
                        continue # 无优化空间

                    eta = 2 * K_full[i, j] - K_full[i, i] - K_full[j, j] #计算二阶导数项 η
                    if eta >= 0:
                        continue # 跳过（理论上 η < 0 才有最大值）

                    alpha_j_new = alphas[j] - y[j] * (E_i - E_j) / eta
                    alpha_j_new = np.clip(alpha_j_new, L, H)     # 限制在一个指定的范围内
                    alpha_i_new = alphas[i] + y[i] * y[j] * (alphas[j] - alpha_j_new)

                    if abs(alpha_i_new - alphas[i]) < 1e-5:
                        continue

                    b1 = b - E_i - y[i] * (alpha_i_new - alphas[i]) * K_full[i, i] - y[j] * (alpha_j_new - alphas[j]) * \
                         K_full[i, j]
                    b2 = b - E_j - y[i] * (alpha_i_new - alphas[i]) * K_full[i, j] - y[j] * (alpha_j_new - alphas[j]) * \
                         K_full[j, j]
                    if 0 < alpha_i_new < self.C:
                        b = b1
                    elif 0 < alpha_j_new < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    alphas[i] = alpha_i_new
                    alphas[j] = alpha_j_new
                    changed = True
            if not changed:
                break

        self.b = b
        support_idx = np.where(alphas > 1e-5)[0]
        self.alphas = alphas[support_idx]
        self.support_vectors = X[support_idx]
        self.support_vector_labels = y[support_idx]

    def _predict_single(self, x):
        total = 0.0
        for i in range(len(self.support_vectors)):
            total += self.alphas[i] * self.support_vector_labels[i] * self._kernel(x.reshape(1, -1),
                                                                                   self.support_vectors[i].reshape(1,
                                                                                                                   -1))
        return total + self.b

    def predict(self, X):
        preds = []
        for x in X:
            val = self._predict_single(x)
            # 确保 val 是标量
            if hasattr(val, 'item'):
                val = val.item()  # 转为 Python float
            preds.append(val)
        return np.array(preds)  # shape: (n_samples,)