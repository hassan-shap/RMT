# +
import numpy as np
from math import pi, sqrt, tanh
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from ipywidgets import interact
from os import path

import plotly.graph_objects as go
import pandas as pd

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

Nrep=20
r=2 # local Hilbert space dim
symm=1

Labc= 16
Npt_sw = np.arange(1,int((Labc)/2)+1)
# Npt_sw = [int((Labc)/2)]
Lc_sw=np.arange(Labc-1,1,-1)
# Lc_sw=np.arange(6,1,-1)


for Npt in Npt_sw:
    print(Npt)
    for Lc in Lc_sw:
        Nc=r**Lc
        Lab = Labc-Lc
        Nab = r**Lab
        print("Lc: ", Lc)


        if symm==1:
            X=np.zeros((Nab,Nc), dtype=np.complex128)
            s_ab=Hilbertspace_Zr(Lab,r)
            i_ab=[]
            ### indices for occupation numbers
            for i_r in range(Lab+1):
                i_ab.append(np.argwhere(np.sum(s_ab,axis=1)==i_r)[:,0])

            s_c=Hilbertspace_Zr(Lc,r)
            i_c=[]
            for i_r in range(Lc+1):
                i_c.append(np.argwhere(np.sum(s_c,axis=1)==i_r)[:,0])        

        ln_vals=np.zeros((Nrep,Lab+1))
        mi_vals=np.zeros((Nrep,Lab+1))

        t_timer= time.time()
        for i_r in range(Nrep):
            print(i_r)
            if symm==1:
                for i_pt in range(min(Npt,Lab)+1):
                    if (Npt-i_pt) <= Lc:
                        X[np.ix_(i_ab[i_pt],i_c[Npt-i_pt])]=np.random.randn(len(i_ab[i_pt]),len(i_c[Npt-i_pt]))+ 1j*np.random.randn(len(i_ab[i_pt]),len(i_c[Npt-i_pt]))
            else:
            #### no symmetry
                X=np.random.randn(Nab,Nc)+1j*np.random.randn(Nab,Nc)

            mat=np.dot(X,np.matrix(X).H)
            rho= mat / np.trace(mat)
            for La in range(Lab+1):
                Lb = Labc-Lc-La
                ln_vals[i_r,La]= logneg(rho,dims=[r]*(Lab),sysa=range(La))
                mi_vals[i_r,La]= mutual_information(rho,dims=[r]*Lab, sysa=range(La))
    #         elapsed = time.time() - t_timer
    #         print("density: ", elapsed)

    #         t_timer= time.time()

    #         psi = normalize(np.reshape(X,[Nab*Nc,1]))
    #         for La in range(0,Lab+1):
    #             Lb = Labc-Lc-La
    #             ln_vals2[i_r,La]= logneg_subsys_approx(psi,dims=[r]*(Labc),sysa=range(La),sysb=range(La,La+Lb))

    #         elapsed = time.time() - t_timer
    #         print("state: ", elapsed)

        out_dir = 'data/' 
        for La in range(Lab+1):
            Lb = Labc-Lc-La
            if symm==1:
                f1= 'LN_U1_%d_Labc_%d_%d_%d_symm.npz' % (Npt,La,Lb,Lc)
            else:
                f1= 'LN_Labc_%d_%d_%d.npz' % (La,Lb,Lc)
            print(f1+' was saved!')
            fname = out_dir+f1
            np.savez(fname, ln_vals=ln_vals[:,La], mi_vals=mi_vals[:,La], Nrep=Nrep)
        elapsed = time.time() - t_timer
        print("Finished, elapsed time = %.2f " % (elapsed)+ "sec")
    #     print(ln_vals)
    #     print(ln_vals2)

# -


