import numpy as np

A = np.array([[0,1],[1,1],[1,0]])
#对其进行SVD分解
U,S,Vt = np.linalg.svd(A,full_matrices=True)
print(U,S,Vt)
print('\n')

#由U,S,Vt还原A
A = np.dot(U[:,:2]*S,Vt)
print(A)

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random

X = sparse_random(100,100,density=0.01,format='csr',random_state=42)

svd = TruncatedSVD(n_components=5,n_iter=7,random_state=42)
svd.fit(X)

print(f'\n{svd.singular_values_}')