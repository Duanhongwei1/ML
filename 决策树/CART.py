class TreeNode:
    def __init__(self,feature_ix,threshold=None,leaf_value=None,left_branch=None,right_branch=None):
        #特征索引
        self.feature_ix=feature_ix
        #特征的划分阈值
        self.threshold=threshold
        #叶子节点的取值
        self.leaf_value=leaf_value
        #左子树
        self.left_branch=left_branch
        #右子树
        self.right_branch=right_branch

import numpy as np
from utils import feature_split,calculate_gini

##定义二叉决策树
class BinaryDecisionTree:
    #初始化参数
    def __init__(self,min_samples_split=3,min_gini_impurity=999,max_depth=float("inf"),loss=None):
        #根节点
        self.root = None
        #节点的最小分裂样本数
        self.min_samples_split = min_samples_split
        #结点的基尼不纯度
        self.min_gini_impurity = min_gini_impurity
        #树的最大深度
        self.max_depth = max_depth
        #基尼不纯度计算函数
        self.gini_impurity_calculation = None
        #叶子节点值预测函数
        self._leaf_value_calculation = None
        #损失函数
        self.criterion_func = None

    #决策树拟合函数
    def fit(self,X,y,loss=None):
        #递归构建决策树
        self.root = self._construct_tree(X,y)
        self.loss = None

    #决策树构建函数
    def _construct_tree(self,X,y,current_depth=0):
        #初始化最小基尼不纯度
        init_gini_impurity = 999
        #初始化最优特征索引和阈值
        best_criteria = None
        #初始化数据子集
        best_sets = None

        #合并输入和标签
        Xy = np.concatenate((X,y),axis=1)
        #获取样本数和特征数
        m,n = X.shape

        #设定决策树构建条件
        #训练样本量大于节点最小分裂样本数且当前树深度小于最大深度
        if m>=self.min_samples_split and current_depth<=self.max_depth:
            #遍历计算每个特征的基尼不纯度
            for f_i in range(n):
                #获取第i特征的所有取值
                f_values = np.expand_dims(X[:,f_i],axis=1)
                #获取第i个特征的唯一取值
                unique_values = np.unique(f_values)

                #遍历取值并寻找最佳特征分裂阈值
                for threshold in unique_values:
                    # 特征节点二叉分裂
                    Xy1, Xy2 = feature_split(Xy, f_i, threshold)
                    # 如果分裂后的子集大小都不为0
                    if len(Xy1) != 0 and len(Xy2) != 0 :
                        # 获取两个子集的标签值
                        y1 = Xy1[:, n:]
                        y2 = Xy2[:, n:]
                        # 计算基尼不纯度
                        impurity = calculate_gini(y,y1,y2)
                        # 获取最小基尼不纯度
                        # 最佳特征索引和分裂阈值
                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {"feature_i": f_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n],  # X的左子树
                                "lefty": Xy1[:, n:],  # y的左子树
                                "rightX": Xy2[:, :n],  # X的右子树
                                "righty": Xy2[:, n:]  # y的右子树
                            }
         #如果计算的最小基尼不纯度小于设定的最小基尼不纯度
        if init_gini_impurity < self.min_gini_impurity:
            #分别构建左右子树
            left_branch = self._construct_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            right_branch = self._construct_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return TreeNode(feature_ix=best_criteria["f_i"],threshold=best_criteria["threshold"],left_branch=left_branch,right_branch=right_branch)
        #计算叶子节点的取值
        left_value = self.leaf_value_calc(y)
        return TreeNode(leaf_value=left_value)

    #定义二叉树的预测函数
    def predict_value(self,x,tree=None):
        if tree is None:
            tree = self.root
        #如果叶子节点已有值，则直接返回已有值
        if tree.leaf_value is not None:
            return tree.leaf_value
        #选择特征并获取特征值
        feature_value = x[tree.feature_ix]
        #判断落入左子树还是右子树
        branch = tree.right_branch
        if feature_value >= tree.threshold:
            branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch
        #测试子集
        return self.predict_value(x, branch)
    #数据集预测函数
    def predict(self,X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred