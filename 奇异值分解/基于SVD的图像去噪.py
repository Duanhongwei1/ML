#SVD图像压缩
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def restore(u,s,v,K):
    m,n = len(u),len(v[0])
    a = np.zeros((m,n))
    for k in range(K):
        uK = u[:,k].reshape(m,1)
        vK = v[k].reshape(1,n)
        a += s[k] * np.dot(uK,vK)
    a = a.clip(0,255)
    return np.rint(a).astype('uint8')


img = np.array(Image.open('louwill.jpg', 'r'))
u_r,s_r,v_r = np.linalg.svd(img[:,:,0])
u_g,s_g,v_g = np.linalg.svd(img[:,:,1])
u_b,s_b,v_b = np.linalg.svd(img[:,:,2])

K = 50
output_path = r'svd_pic'
if not os.path.exists(output_path):
    os.makedirs(output_path)
for k in tqdm(range(1,K+1)):
    R = restore(u_r,s_r,v_r,k)
    G = restore(u_g,s_g,v_g,k)
    B = restore(u_b,s_b,v_b,k)
    I = np.stack((R,G,B),axis=2)
    Image.fromarray(I).save('%s\\svd_%d.jpg'%(output_path,k))