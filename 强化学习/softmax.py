import numpy as np

def R(k):
    success_probabilities = [0.4, 0.2, 0.3, 0.4, 0.5]  # 各动作的成功概率
    return 1 if np.random.rand() < success_probabilities[k] else 0

def softmax(K,R,T,tao):
    r = 0
    Q = np.zeros(K)
    count = np.zeros(K)
    for t in range(T):
        k = np.random.randint(K)
        v = R(k)
        r += v
        Q[k] = (Q[k]*count[k] + v)/(count[k]+1)
        count[k] += 1
    return r,Q
tao = 0.01
r ,Q= softmax(2,R,1000,tao)
print('实验总回报:',r)
print('最终Q值:',Q)