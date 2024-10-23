#定义欧氏距离
import numpy as np

def euclidean_distance(x,y):
    #初始化距离
    distance = 0
    for i in range(len(x)):
        distance += pow(x[i]-y[i],2)
    return np.sqrt(distance)

#定义质心初始化函数
def centroids_init(X,k):
    m,n = X.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        centroid = X[np.random.choice(range(m))]
        centroids[i] = centroid
    return centroids

#根据质心和距离判断所属质心索引
def closest_centroid(x,centroids):
    #初始化最近索引和最近距离
    closest_i,closest_dist = 0,float('inf')
    #遍历质心矩阵
    for i ,centroid in enumerate(centroids):
        #计算欧式距离
        distance = euclidean_distance(x,centroid)
        #根据欧式距离判断最近距离和最近索引
        if distance < closest_dist:
            closest_i = i
            closest_dist = distance
    return closest_i

#为每个样本分配簇
def bulid_clusters(centroids,k,X):
    #初始化簇列表
    clusters = [[] for _ in range(k)]
    #遍历训练样本
    for x_i,x in enumerate(X):
        #获取样本所属最近质心的索引
        centroid_i = closest_centroid(x,centroids)
        #将当前样本添加到对应质心的簇中
        clusters[centroid_i].append(x_i)
    return clusters

#计算当前质心
def calculate_centroids(clusters,k,X):
    #特征数
    n = X.shape[1]
    #初始化质心矩阵、大小为质心个数X特征数
    centroids = np.zeros((k,n))
    #遍历当前簇
    for i,cluster in enumerate(clusters):
        #计算每个簇的均值作为新的质心
        centroid = np.mean(X[cluster],axis=0)
        #将质心向量分配给质心矩阵
        centroids[i] = centroid
    return centroids

#获取样本所属的聚类类别
def get_cluster_labels(clusters,X):
    #预测结果初始化
    y_pred = np.zeros(X.shape[0])
    #遍历聚类簇
    for cluster_i,cluster in enumerate(clusters):
        #遍历当前簇
        for sample_i in cluster:
            #为每个样本分配类别簇
            y_pred[sample_i] = cluster_i
    return y_pred

#封装
def kmeans(X,k,max_iterations):
    #初始化质心
    centroids = centroids_init(X,k)
    #遍历迭代求解
    for _ in range(max_iterations):
        #根据当前质心进行聚类
        clusters = bulid_clusters(centroids,k,X)
        #保存当前质心
        cur_centroids = centroids
        #根据聚类结果计算新的质心
        centroids = calculate_centroids(clusters,k,X)
        #设定收敛条件为质心是否发生变化
        diff = centroids - cur_centroids
        if not diff.any():
            break
    #返回最终的聚类标签
    return get_cluster_labels(clusters,X)

import matplotlib.pyplot as plt
if __name__ == '__main__':
    X = np.random.randn(200,2)
    labels = kmeans(X,2,10)
    print(labels)
    plt.scatter(X[:,0],X[:,1],c=labels,s=50)
    plt.show()