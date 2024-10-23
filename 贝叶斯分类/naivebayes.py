import pandas as pd
import numpy as np

def nb_fit(X,y):
    #标签类别
    classes = y[y.columns[0]].unique()
    #标签类别统计
    class_count = y[y.columns[0]].value_counts()
    #极大似然估计:类先验概率
    class_prior = class_count/len(y)
    #类条件概率:字典初始化
    prior_condition_prob = dict()
    #遍历计算类条件概率
    for col in X.columns:
        for j in classes:
            p_x_y = X[(y==j).values][col].value_counts()
            for i in p_x_y.index:
                prior_condition_prob[(col,i,j)] = p_x_y[i]/class_count[j]
    return classes,class_prior,prior_condition_prob





def nb_predict(X_test, classes, class_prior, prior_condition_prob):
    res = []
    for c in classes:
        # 获取当前类的先验概率
        p_y = class_prior[c]
        # 初始化类条件概率
        p_x_y = 1.0

        # 遍历每个特征及其值
        for feature, value in X_test.iloc[0].items():
            # 构建条件概率字典的键
            key = (feature, value, c)

            # 检查该键是否存在于条件概率字典中
            if key in prior_condition_prob:
                p_x_y *= prior_condition_prob[key]  # 乘以条件概率 P(x_i|c)
            else:
                # 如果数据集中没有出现这种组合，则使用一个非常小的概率值
                p_x_y *= 1e-8  # 或者其他很小的值

        # 计算后验概率 P(c|x) 的分子部分
        res.append(p_y * p_x_y)

    # 返回具有最大后验概率的类别
    return classes[np.argmax(res)]