import numpy as np

def em(data,thetas,max_iter=50,eps=1e-3):
    #初始化似然函数值
    ll_old = 0
    for i in range(max_iter):
        #E步:求隐变量分布
        #对数似然
        log_like = np.array([np.sum(data*np.log(theta),axis=1)for theta in thetas])
        #似然
        like = np.exp(log_like)
        #求隐变量分布
        ws = like/like.sum()
        #概率加权
        vs = np.array([w[:,None]*data for w in ws])
        #M步:更新参数值
        thetas = np.array([v.sum(0)/v.sum() for v in vs])
        #更新似然函数
        ll_new = np.sum([w*l for w,l in zip(ws,log_like)])
        print('iteration:%d'%(i+1))
        print('theta_B = %.2f,theta_C = %.2f,ll = %.2f'%(thetas[0,0],thetas[1,0],ll_new))
        #满足迭代条件即推出迭代
        if np.abs(ll_new-ll_old) < eps:
            break
        ll_old = ll_new
    return thetas