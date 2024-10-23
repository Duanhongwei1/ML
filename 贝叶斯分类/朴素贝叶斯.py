import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.classes = None  # 存储所有类别
        self.priors = {}  # 存储每个类别的先验概率
        self.likelihoods = {}  # 存储条件概率

    def fit(self, X, y):
        '''训练模型
        X: 输入特征（DataFrame）
        y: 目标标签（Series）
        '''
        self.classes = np.unique(y)
        total_samples = len(y)

        for c in self.classes:
            # 获取当前类别的样本
            X_c = X[y == c]
            # 计算先验概率 P(C)
            self.priors[c] = len(X_c) / total_samples
            # 计算每个特征下的条件概率 P(X|C)
            self.likelihoods[c] = {}
            for feature in X.columns:
                self.likelihoods[c][feature] = X_c[feature].value_counts(normalize=True).to_dict()

    def predict(self, X):
        '''预测输入实例的类别
        X: 输入特征（DataFrame）
        '''
        results = []
        for i in range(len(X)):
            posteriors = {}
            for c in self.classes:
                # 初始化后验概率为先验概率
                post_prob = np.log(self.priors[c])
                for feature in X.columns:
                    # 获取每个特征的条件概率 P(X|C)
                    feature_value = X.iloc[i][feature]
                    if feature_value in self.likelihoods[c][feature]:
                        post_prob += np.log(self.likelihoods[c][feature][feature_value])
                    else:
                        post_prob += np.log(1e-6)  # 防止0概率的出现，进行拉普拉斯平滑
                posteriors[c] = post_prob
            # 选择后验概率最大的类别
            results.append(max(posteriors, key=posteriors.get))
        return results
