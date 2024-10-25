import numpy as np
class HMM:
    def __init__(self,N,M,pi=None,A=None,B=None):
        #可能的状态数
        self.N = N
        #可能的观测数
        self.M = M
        #初始状态概率向量
        self.pi = pi
        #状态转移概率矩阵
        self.A = A
        #观测概率矩阵
        self.B = B

    #根据给定的概率分布随机返回数据
    def rdistribution(self,dist):
        r = np.random.rand()
        for ix ,p in enumerate(dist):
            if r < p:
                return ix
            r -= p
    #生成HMM观测序列
    def generate(self,T):
        #根据初始概率分布生成第一个状态
        i = self.rdistribution(self.pi)
        #生成第一个观测数据
        o = self.rdistribution(self.B[i])
        observed_data = [o]
        #遍历生成后续的状态和观测数据
        for _ in range(T-1):
            i = self.rdistribution(self.A[i])
            o = self.rdistribution(self.B[i])
            observed_data.append(o)
        return observed_data

    ###前向算法计算条件概率
    def prob_calc(self,O):
        '''
        Args:
            O: 观测序列
        Returns:条件概率
        '''
        #初始值
        alpha = self.pi * self.B[:,O[0]]
        #递推
        for o in O[1:]:
            alpha_next = np.empty(4)
            for j in range(4):
                alpha_next[j] = np.sum(self.A[:,j]*alpha*self.B[j,o])
            alpha = alpha_next
        return alpha.sum()