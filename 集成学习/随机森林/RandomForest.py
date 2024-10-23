#定义自助抽样函数
import numpy as np
from sklearn.metrics import accuracy_score


def bootstrap_sampling(X,y,n_estimators):
    '''

    Args:
        X:训练样本
        y: 训练样本标签
        n_estimators: 树的棵数

    Returns:
        sampling_subsets:抽样子集
    '''
    X_y = np.concatenate([X,y.reshape(-1,1)],axis=1)
    np.random.shuffle(X_y)
    n_samples = X_y.shape[0]
    #初始化抽样子集列表
    sampling_subsets = []
    for _ in range(n_estimators):
        #第一个随机性抽样
        idx1 = np.random.choice(n_samples,n_samples,replace=True)
        bootstrap_Xy = X_y[idx1,:]
        bootstrap_X = bootstrap_Xy[:,:-1]
        bootstrap_y = bootstrap_Xy[:,-1]
        sampling_subsets.append([bootstrap_X,bootstrap_y])
    return sampling_subsets
#给定输入输出数据集和决策树棵树，通过随机抽样的方式构造多个抽样子集

#构造随机森林
from cart import ClassificationTree
#树的棵数
n_estimators = 10
#初始化随机森林所包含的数列表
trees = []
#基于决策树构建森林
for _ in range(n_estimators):
    tree = ClassificationTree(min_samples_split=2,min_gini_impurity=999,max_depth=3)
    trees.append(tree)


###定义随机森林类
class RandomForest:
    def __init__(self,n_estimators=100,min_samples_split=2,min_gini=0,max_depth=float('inf'),max_features=None):
        #树的棵数
        self.n_estimators = n_estimators
        #树最小分裂样本数
        self.min_samples_split = min_samples_split
        #最小基尼不纯度
        self.min_gini = min_gini
        #树的最大深度
        self.max_depth = max_depth
        #所使用最大特征数
        self.max_features = max_features
        self.trees = []

        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split,min_gini_impurity=self.min_gini,max_depth=self.max_depth)
            self.trees.append(tree)
    def bootstrap_sampling(self,X,y):
        X_y = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        np.random.shuffle(X_y)
        n_samples = X_y.shape[0]
        # 初始化抽样子集列表
        sampling_subsets = []
        for _ in range(n_estimators):
            # 第一个随机性抽样
            idx1 = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_Xy = X_y[idx1, :]
            bootstrap_X = bootstrap_Xy[:, :-1]
            bootstrap_y = bootstrap_Xy[:, -1]
            sampling_subsets.append([bootstrap_X, bootstrap_y])
        return sampling_subsets
    def fit(self,X,y):
        #对森林中每棵树训练一个双随机抽样子集
        sub_sets = self.bootstrap_sampling(X,y)
        n_features = X.shape[1]

        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            sub_X,sub_y = sub_sets[i]
            idx2 = np.random.choice(n_features,self.max_features,replace=True)
            sub_X = sub_X[:,idx2]
            self.trees[i].fit(sub_X,sub_y)
            self.trees[i].feature_indices = idx2
            print('The {}th tree is trained done...'.format(i+1))

    def predict(self,X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices[i]
            sub_X = X[:,idx]
            y_pred = self.trees[i].predict(sub_X)
            y_preds.append(y_pred)
        y_preds = np.array(y_preds).T
        res = []
        for j in y_preds:
            res.append(np.bincount(j.astype('int')).argmax())
        return res



