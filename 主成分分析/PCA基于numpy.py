import numpy as np

class PCA:
    #定义协方差矩阵计算方法
    def calc_cov(self,X):
        #样本量:
        m = X.shape[0]
        #数据标准化
        X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
        return 1 / m * np.matmul(X.T,X)
    #PCA算法实现
    def pca(self,X,n_components): #输入为要进行PCA的矩阵和指定的主成分个数
        #计算协方差矩阵
        conv_matrix = self.calc_cov(X)
        #计算协方差矩阵的特征值和特征向量
        eigenvalues,eigenvectors = np.linalg.eig(conv_matrix)    #np.linalg.eig用于求特征值和特征向量
        #对特征值进行排序
        idx = eigenvalues.argsort()[::-1]
        #取最大的前n_component组
        eigenvectors = eigenvectors[:,idx]
        eigenvectors = eigenvectors[:,:n_components]
        #Y = PX 转换
        return np.matmul(X,eigenvectors)




from sklearn import datasets
import matplotlib.pyplot as plt
if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print('X的形状为:',X.shape)
    PCA = PCA()
    X_trans = PCA.pca(X,3)
    print('X_trans的形状为:',X_trans.shape)
    #颜色列表
    colors = ['navy','turquoise','darkorange']
    #绘制不同类别
    for c,i,target_name in zip(colors,[0,1,2],iris.target_names):
        plt.scatter(X_trans[y==i,0],X_trans[y==i,1],color=c,lw=2,label=target_name)

    plt.legend()
    plt.show()