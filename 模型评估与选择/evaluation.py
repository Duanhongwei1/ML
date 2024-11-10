import numpy as np
from sklearn import metrics
def hold_out (data,train_size):   #留出法
    length = int(len(data)*train_size)
    train_data = data[:length]
    test_data = data[length:]
    return train_data,test_data


import numpy as np
from sklearn.metrics import accuracy_score


def k_fold_cross_validation(model, X, y, K=5, metric=accuracy_score):
    """
    使用 K 折交叉验证评估模型。

    参数:
    model: 训练模型 (需实现 fit 和 predict 方法)
    X: 特征数据 (numpy array 或 pandas DataFrame)
    y: 标签数据 (numpy array 或 pandas Series)
    K: 交叉验证的折数 (默认5)
    metric: 评估指标 (默认 accuracy_score)

    返回:
    每折的评分列表和平均评分
    """
    num_samples = len(y)
    fold_size = num_samples // K
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    scores = []

    for k in range(K):
        start, end = k * fold_size, (k + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        # 训练模型
        model.fit(X_train, y_train)

        # 计算验证集上的得分
        y_pred = model.predict(X_val)
        score = metric(y_val, y_pred)
        scores.append(score)
        print(f"第{k + 1}折验证集得分: {score:.4f}")

    # 计算平均得分
    average_score = np.mean(scores)
    print(f"{K} 折交叉验证的平均得分: {average_score:.4f}")
    return scores, average_score


