import numpy as np
def R(k):
    success_probabilities = [0.4, 0.2, 0.3, 0.4, 0.5]  # 各动作的成功概率
    return 1 if np.random.rand() < success_probabilities[k] else 0

def e_greed(K,R,T,epsilon):
    r = 0
    Q = np.zeros(K)
    count = np.zeros(K)
    for t in range(T):
        if np.random.rand() < epsilon:  #当随机数小于epsilon时，进行的是探索，即随机选择一个动作
            k = np.random.randint(0,K)
        else:
            k = np.argmax(Q)      #当随机数大于等于epsilon时，进行的是利用，即选择Q最大的动作
        v = R(k)
        r += v
        Q[k] = (Q[k]*count[k] + v)/(count[k] + 1)
        count[k] += 1
    return r,Q

epsilon = 0.01
r ,Q= e_greed(2,R,1000,epsilon)
print('实验总回报:',r)
print('最终Q值:',Q)