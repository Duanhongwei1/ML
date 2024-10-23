#以决策树为例定义一个基分类器
class DecisionStump:
    def __init__(self):
        #基于划分阈值决定样本分类为1还是-1
        self.label = 1
        #特征索引
        self.feature_index = None
        #特征划分阈值
        self.threshold = None
        #知识分类准确率的值
        self.alpha = None


#AdaBoost拟合函数
import numpy as np
##AdaBoost算法拟合过程
def fit(X,y,n_estimators):
    '''
    输入:
    X:训练输入
    y:训练输出
    n_estimators:基分类器个数
    输出:
    estimators:包含所有基分类器的列表
    '''

    m,n = X.shape
    #1.初始化权重分布为均匀分布1/N
    w = np.full(m,(1/m))
    #初始化基分类器列表
    estimators = []

    for _ in range(n_estimators):
        #2.a 训练一个弱分类器
        estimator = DecisionStump()
        #设定一个最小化误差
        min_error = float('inf')
        #遍历数据集特征:根据最小分类误差率选择最优特征
        for i in range(n):
            #获取特征值
            values = np.expand_dims(X[:,i],axis=1)
            #特征取值去重
            unique_values = np.unique(values)
            #尝试将每一个特征值作为分类阈值
            for threshold in unique_values:
                p = 1
                #初始化所有预测值为1
                pred = np.ones(np.shape(y))
                #小于分类阈值的预测值为-1
                pred[X[:,i]<threshold] = -1
                #计算分类误差率
                error = sum(w[y != pred])
                #如果分类误差率大于0.5，则进行正负预测翻转
                if error > 0.5 :
                    error = 1 - error
                    p = -1
                #一旦获得最小误差,则保存相关参数配置
                if error < min_error:
                    estimator.label = p
                    estimator.threshold = threshold
                    estimator.feature_index = i
                    min_error = error
        #计算基分类器的权重
        estimator.alpha = 0.5 * np.log((1.0-min_error)/(min_error+1e-9))
        #初始化所有预测值为1
        preds = np.ones(np.shape(y))
        #获取所有小于阈值的负类索引
        negative_idx = (estimator.label*X[:,estimator.feature_index]<estimator.label*estimator.threshold)
        #将负类设为-1
        preds[negative_idx] = -1
        #更新样本权重
        w *= np.exp(-estimator.alpha*y*preds)
        w /= np.sum(w)
        #保存该弱分类器
        estimators.append(estimator)
#定义AdaBoost预测函数
def predict(X,estimators):
    '''
    Args:
        X:预测输入
        estimators:包含所有基分类器的列表
    Returns:预测输出
    '''
    m = len(X)
    y_pred = np.zeros((m,1))
    #计算每个基分类器的预测值
    for estimator in estimators:
        #初始化所有预测值为1
        predictions = np.ones(np.shape(y_pred))
        #获取所有小于阈值的负类索引
        negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
        #将负类设为-1
        predictions[negative_idx] = -1
        #对每个基分类器的预测结果进行加权
        y_pred += estimator.alpha * predictions

    y_pred += np.sign(y_pred).flatten()
    return y_pred




#定义Adaboost类
class Adaboost:
    def __init__(self,n_estimators=5):
        self.n_estimators = n_estimators
    #Adaboost拟合算法
    def fit(self,X,y):
        m, n = X.shape
        # 1.初始化权重分布为均匀分布1/N
        w = np.full(m, (1 / m))
        # 初始化基分类器列表
        self.estimators = []

        for _ in range(self.n_estimators):
            # 2.a 训练一个弱分类器
            estimator = DecisionStump()
            # 设定一个最小化误差
            min_error = float('inf')
            # 遍历数据集特征:根据最小分类误差率选择最优特征
            for i in range(n):
                # 获取特征值
                values = np.expand_dims(X[:, i], axis=1)
                # 特征取值去重
                unique_values = np.unique(values)
                # 尝试将每一个特征值作为分类阈值
                for threshold in unique_values:
                    p = 1
                    # 初始化所有预测值为1
                    pred = np.ones(np.shape(y))
                    # 小于分类阈值的预测值为-1
                    pred[X[:, i] < threshold] = -1
                    # 计算分类误差率
                    error = sum(w[y != pred])
                    # 如果分类误差率大于0.5，则进行正负预测翻转
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    # 一旦获得最小误差,则保存相关参数配置
                    if error < min_error:
                        estimator.label = p
                        estimator.threshold = threshold
                        estimator.feature_index = i
                        min_error = error
            # 计算基分类器的权重
            estimator.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-9))
            # 初始化所有预测值为1
            preds = np.ones(np.shape(y))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
            # 将负类设为-1
            preds[negative_idx] = -1
            # 更新样本权重
            w *= np.exp(-estimator.alpha * y * preds)
            w /= np.sum(w)
            # 保存该弱分类器
            self.estimators.append(estimator)

    def predict(self,X):
        m = len(X)
        y_pred = np.zeros((m, 1))
        # 计算每个基分类器的预测值
        for estimator in self.estimators:
            # 初始化所有预测值为1
            predictions = np.ones(np.shape(y_pred))
            # 获取所有小于阈值的负类索引
            negative_idx = (estimator.label * X[:, estimator.feature_index] < estimator.label * estimator.threshold)
            # 将负类设为-1
            predictions[negative_idx] = -1
            # 对每个基分类器的预测结果进行加权
            y_pred += estimator.alpha * predictions

        y_pred = np.sign(y_pred).flatten()
        return y_pred
