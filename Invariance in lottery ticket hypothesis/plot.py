import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser as AP


SIZE = 12 

plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title




p = AP()
p.add_argument('--npy', type=str, help='Enter numpy file name to load')
p = p.parse_args()

coeff = np.load(p.npy)
coeff = torch.tensor(coeff)

lst = np.arange(0.1,1,0.1)

for i in range(len(lst)):
	sig=[]
	for j in range(coeff.shape[0]):
		a=torch.zeros(coeff[j].shape[0]).long()
		b=torch.arange(0, coeff[j].shape[0])
		c=torch.where(((coeff[j] > -lst[i]) & (coeff[j] < lst[i])),a,b)
		sig.append(torch.sum(c != 0).cpu().numpy())
	sig = np.array(sig)
	plt.plot(sig,label= lst[i])
	plt.legend()
plt.xlabel('iterations')
plt.ylabel('Significant zero components')
plt.close()

for i in range(coeff.shape[0]):
	a=torch.zeros(coeff[i].shape[0]).long()
	b=torch.arange(0, coeff[i].shape[0])
	c=torch.where(((coeff[i] > -0.1) & (coeff[i] < 0.1)),b,a)
	z = torch.zeros(coeff[i].shape[0]).fill_(0)
	z[torch.nonzero(c)] = coeff[i][torch.nonzero(c)]
	plt.plot(z)
plt.xlabel('Dimension')
plt.ylabel('Coefficient')
p.npy = p.npy[:-4]+'.png'
plt.savefig(p.npy, format='png')
