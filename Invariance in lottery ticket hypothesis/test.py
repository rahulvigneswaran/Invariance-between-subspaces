import numpy as np
a = np.random.rand(10,10)
u,s,v = np.linalg.svd(a)

b = a+0.1*np.random.rand(10,10)
uu,ss,vv = np.linalg.svd(b)
'''
print np.linalg.norm(np.matmul(u[:,:5],u[:,-5:].transpose()) -np.matmul(uu[:,:5],uu[:,-5:].transpose()),1)

uu,ss,vv = np.linalg.svd(np.matmul(a,b))
print(ss)
uu,ss,vv = np.linalg.svd(np.matmul(b,a))
print(np.max(np.sin(ss)))
'''
print np.sqrt(5-np.sum(np.sum(np.matmul(u[:,:5].transpose(), uu[:,:5]))))
