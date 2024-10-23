#根据数据集和指定特征定义数据集划分函数
import pandas as pd
import numpy as np
from math import log

#信息熵计算
def entropy(ele):
    '''输入:ele 包含类别取值的列表
        输出:信息熵值'''

    #计算列表中取值的概率分布
    probs = [ele.count(i)/len(ele)for i in set(ele)]
    #计算信息熵
    entropy = -sum([prob*log(prob,2)for prob in probs])
    return entropy


# 定义基尼指数计算函数
def gini(nums):
    '''nums:输入 包含类别取值的列表'''
    probs = [nums.count(i) / len(nums) for i in set(nums)]

    # 计算基尼指数
    gini = sum([p * (1 - p) for p in probs])
    return gini


def df_split(df,col):
    '''输入:
        df 待划分的训练数据
        col 划分数据的依据特征
        输出：
        res_dict 划分后的字典'''
    #获取依据特征的不同取值
    unique_col_val = df[col].unique()
    #创建划分结果的数据字典
    res_dict = {elem:pd.DataFrame for elem in unique_col_val}
    #根据特征取值进行划分
    for key in res_dict.keys():
        res_dict[key] = df[:][df[col]==key]
    return res_dict

#选择最有特征
def choose_best_feature(df,label):
    '''输入:
        df 待划分的训练数据
        label 训练标签
        输出
        max_value:最大信息增益值
        best_feature 最优特征
        max_splited 根据最优特征划分后的数据字典'''

    #计算训练标签的信息熵
    entropy_D = entropy(df[label].tolist())
    #特征集
    cols = [col for col in df.columns if col not in [label]]
    #初始化最大信息增益值，最优特征和划分后的数据集
    max_value , best_feature = -999, None
    max_splited = None

    #遍历特征并根据特征取值进行划分
    for col in cols:
    #根据当前特征划分后的数据集
        splited_set = df_split(df,col)
    #初始化经验条件熵
        entropy_DA = 0
        for subset_col ,subset in splited_set.items():
            #计算划分后的数据子集的标签信息熵
            entropy_Di = entropy(subset[label].tolist())
            #计算当前特征的经验条件熵
            entropy_DA += len(subset)/len(df) * entropy_Di
    #计算当前特征的信息增益
        info_gain = entropy_D - entropy_DA
    #获取最大信息增益
        if info_gain > max_value:
            max_value ,best_feature = info_gain,col
            max_splited = splited_set
    return max_value,best_feature,max_splited

#构建ID3决策树
class ID3Tree:
    #定义决策树节点类
    class TreeNode:
        #定义树节点
        def __init__(self,name):
            self.name = name
            self.connections = {}
        #定义树连接
        def connect(self,label,node):
            self.connections[label] = node
    #定义全局变量，包括数据集，特征集，标签和根节点
    def __init__(self,df,label):
        self.columns = df.columns
        self.df = df
        self.label = label
        self.root = self.TreeNode("Root")
    #构建树的调用
    def construct_tree(self):
        self.construct(self.root,"",self.df,self.columns)
    #决策树构建方法
    def construct(self,parent_node,parent_label,sub_df,columns):
        #选择最优特征
        max_value,best_feature,max_splited = choose_best_feature(sub_df[columns],self.label)
        #如果选不到最优特征，则构造单节点树
        if not best_feature:
            node = self.TreeNode(sub_df[self.label].iloc[0])
            parent_node.connect(parent_label,node)
            return
        #根据最优特征及子节点构建树
        node = self.TreeNode(best_feature)
        parent_node.connect(parent_label,node)
        #以A-Ag为新的特征集
        new_columns = [col for col in columns if col != best_feature]
        #递归的构造决策树
        for splited_value ,splited_data in max_splited.items():
            self.construct(node,splited_value,splited_data,new_columns)

    