#定义辅助函数
import numpy as np

#定义sign符号函数
def sign(x,w,b):
    '''输入：
    x：输入实例
    w：权重系数
    b：偏置参数
    输出：符号函数值'''
    return np.dot(x,w)+b

#定义参数初始化函数
def initialize_parameters(dim):
    '''输入：
    dim：输入数据维度
    输出：
    w：初始化后的权重系数
    b：初始化后的偏置系数'''
    w = np.zeros(dim)
    b = 0.0
    return w,b


#定义感知机训练过程
def perceptron_train(X_train,y_train,learning_rate):
    #参数初始化
    w,b = initialize_parameters(X_train.shape[1])
    #初始化误分类状态
    is_wrong = False
    #当存在误分类点时
    while not is_wrong:
        #初始化误分类点计数
        wrong_count = 0
        for i in range(len(X_train)):
            X = X_train[i]
            y = y_train[i]
            #如果存在误分类点
            if y*sign(X,w,b) <= 0:
                #更新参数
                w = w + learning_rate*np.dot(y,X)
                b = b + learning_rate*y
                #误分类点+1
                wrong_count += 1
        #直到没有误分类点
        if wrong_count == 0:
            is_wrong = True
            print("there is no missclassification!")
        #保存更新后的参数
        params = {
            'w':w,
            'b':b
        }
    return params