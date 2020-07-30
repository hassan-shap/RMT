#!/tmp/yes/bin python3

import numpy as np
from math import pi, sqrt, tanh, asin
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from ipywidgets import interact
from os import path

from quimb import *
import time

def Hilbertspace_Zr(N,r):

    states=np.zeros((r**N,N),dtype=int)

    if N>0:
        for i_1 in range(r**N):
            num_str=np.base_repr(i_1,base=r)[::-1]
            for i_2 in range(len(num_str)):
                states[i_1,i_2]=int(num_str[i_2])
    else:
        states=[[0]]
        
    return states

Nrep=1
r=4 # local Hilbert space dim

L=14
Lab_sw=np.arange(2,L+1)
# La_sw=np.arange(1,int(L/2)+1)
# La_sw= [1] # range(1,6) 

print('LN_r_%d_L_%d' % (r,L))


dims = [r] * L
N = prod(dims)

s_abc=Hilbertspace_Zr(L,r)

Ne_abc = r**(L-1)

t_timer=time.time()

neg_symm=np.zeros((len(Lab_sw),Nrep))
neg_full=np.zeros((len(Lab_sw),Nrep))
# np.random.seed(1)

for i_l in range(len(Lab_sw)):
    
    Lb=int(Lab_sw[i_l]/2)
    La=Lab_sw[i_l]-Lb
    Lc=L-La-Lb
    print('Lab: ',La+Lb)
    
    Na=r**La
    Nb=r**Lb
    dims_ab = [r] * (La+Lb)
    Nc=r**Lc
    
    if Lc>0:
        Ne_ab = r**(L-2)
    else:
        Ne_ab = r**(L-1)

    i_abc=np.zeros((Ne_abc,r),dtype=int)
    i_c=np.zeros((Ne_ab,r),dtype=int)
    ### indices of r multiples
    for i_Zr in range(r):
        if Lc>0:
            i_abc[:,i_Zr]=np.argwhere(np.mod(np.sum(s_abc[:,:(La+Lb)],axis=1),r)==i_Zr)[:,0]
            i_v=np.argwhere(np.mod(np.sum(s_abc[i_abc[:,i_Zr],(La+Lb):],axis=1),r)==i_Zr)[:,0]
            i_c[:,i_Zr]=i_abc[i_v,i_Zr]
        else:
            i_c[:,i_Zr]=np.argwhere(np.mod(np.sum(s_abc,axis=1),r)==i_Zr)[:,0]

    for i_r in range(Nrep):        
        psi = rand_ket(N)
        neg_full[i_l,i_r]=logneg_subsys_approx(psi, dims=dims, sysa=np.arange(Lc,Lc+La), sysb=np.arange(Lc+La,L))

        vec=np.random.randn(Ne_ab)+ 1j*np.random.randn(Ne_ab)
        psi = np.zeros(N, dtype=np.complex128)
        psi[i_c[:,0]]=vec/np.linalg.norm(vec)

        neg_symm[i_l,i_r]=logneg_subsys_approx(psi, dims=dims, sysa=np.arange(Lc,Lc+La), sysb=np.arange(Lc+La,L))
    
    print('Nrep:', i_r)

elapsed = time.time() - t_timer
print("Finished, quimb elapsed time = %.2f " % (elapsed)+ "sec")

f1= 'LN_r_%d_L_%d.npz' % (r,L)
out_dir = 'data/' 
fname = out_dir+f1
np.savez(fname, neg_symm=neg_symm, neg_full=neg_full, Lab_sw=Lab_sw, Nrep=Nrep)

