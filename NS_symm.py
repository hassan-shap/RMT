#!/tmp/yes/bin python3

import numpy as np
from math import pi, sqrt, tanh
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from ipywidgets import interact
from os import path
#from quimb import *
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

def p_transpose_2(V,Na,Nb):
# partial transpose with respect to subsystem 2
# the basis of NaxNb density matrix is defined by Nb*(i-1)+j,
# i,j=1,2,3 spans the Hilbert space of subsystem 1 and 2 respectively
    U=np.zeros((Na*Nb,Na*Nb), dtype=np.complex128)
    for i_1 in range(Na):
        for i_2 in range(Na):
            U[Nb*i_1:Nb*(i_1+1),Nb*i_2:Nb*(i_2+1)]=np.transpose(V[Nb*i_1:Nb*(i_1+1),Nb*i_2:Nb*(i_2+1)])

    return U


Nrep=100
r=4 # local Hilbert space dim
symm=1

La=3
Lb=La
Na=r**La
Nb=r**Lb

if symm==1:
    f1= 'NS_r_%d_LA_%d_symm' % (r,La+Lb)
else:
    f1= 'NS_r_%d_LA_%d' % (r,La+Lb)
    
print(f1)

if symm==1:
    Nab_r=r**(La+Lb-1)
    s_ab=Hilbertspace_Zr(La+Lb,r)
    i_ab=np.zeros((r**(La+Lb-1),r),dtype=int)
    ### indices of r multiples
    for i_Zr in range(r):
        i_ab[:,i_Zr]=np.argwhere(np.mod(np.sum(s_ab,axis=1),r)==i_Zr)[:,0]
else:
    Nab_r=Na*Nb

# L_sw=range(8,13)
# L_sw=range(12,16)
L_sw=[15]

# v1T=np.zeros((Nrep*Nb*Na,len(L_sw)))
v1T=np.zeros(Nrep*Nb*Na)
# neg=np.zeros((Nrep,len(L_sw)))


t_timer= time.time()
for i_l in range(len(L_sw)):
    L=L_sw[i_l]
    print(L)

    Lc=L-La-Lb
    Nc=r**Lc
    

    if symm==1:
        Nc_r=r**(Lc-1)
        s_c=Hilbertspace_Zr(Lc,r)
        i_c=np.zeros((r**(Lc-1),r),dtype=int)
        ### indices of r multiples
        for i_Zr in range(r):
            i_c[:,i_Zr]=np.argwhere(np.mod(np.sum(s_c,axis=1),r)==i_Zr)[:,0]
    else:
        Nc_r=Nc

    X=np.zeros((Na*Nb,Nc), dtype=np.complex128)

    for i_r in range(Nrep):
        i_r
        if symm==1:
            X[np.ix_(i_ab[:,0],i_c[:,0])]=np.random.randn(Nab_r,Nc_r)+ 1j*np.random.randn(Nab_r,Nc_r)
    #         X[i_ab[:,0],:] = np.random.randn(Ne_ab,Nc)+ 1j*np.random.randn(Ne_ab,Nc)
        else:
        #### no symmetry
            X=np.random.randn(Nab_r,Nc)+1j*np.random.randn(Nab_r,Nc_r)

        mat=np.dot(X,np.matrix(X).H)
        rho= mat / np.trace(mat)

        rT2 = p_transpose_2(rho,Na,Nb)
        l1T=np.linalg.eigvalsh(rT2)
        v1T[i_r*Nb*Na:(i_r+1)*Nb*Na] = Nab_r *l1T
#         v1T[i_r*Nb*Na:(i_r+1)*Nb*Na,i_l] = Nab_r *l1T
#         neg[i_r,i_l]=np.sum(np.abs(l1T))

    if symm==1:
        f1= 'NS_r_%d_LA_%d_L_%d_symm.npz' % (r,La+Lb,L)
    else:
        f1= 'NS_r_%d_LA_%d_L_%d.npz' % (r,La+Lb,L)
    out_dir = 'data/' 
    fname = out_dir+f1
    np.savez(fname, evals=v1T, Nrep=Nrep)
    
elapsed = time.time() - t_timer
print("Finished, elapsed time = %.2f " % (elapsed)+ "sec")

