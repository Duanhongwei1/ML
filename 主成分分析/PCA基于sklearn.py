import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_iris

iris = load_iris()
X,y = iris.data,iris.target

pca = decomposition.PCA(n_components=3)
pca.fit(X)
X_trans = pca.transform(X)
colors = ['navy','turquoise','darkorange']
for c,i,target_name in zip(colors,[0,1,2],iris.target_names):
   plt.scatter(X_trans[y==i,0],X_trans[y==i,1],color=c,lw=2,label=target_name)

plt.legend()
plt.show()