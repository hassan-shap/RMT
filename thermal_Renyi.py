# -*- coding: utf-8 -*-
import numpy as np
import scipy
from math import pi, sqrt, tanh
import matplotlib.pyplot as plt
import time
from os import path

import itertools
from operator import add
from quimb import *
from quimb.linalg.approx_spectral import *
import quimb

r=2 # local Hilbert space dim


def ising_chain(L, g=1.0, h=0.0, cyclic=True,
                sparse=True):

    dims = [r] * L  # shape (n, m)

    # generate tuple of all site coordinates
    # sites = tuple(itertools.product(range(n)))
    sites= tuple(range(L))
    # print(sites)

    # generate neighbouring pairs of coordinates
    def gen_pairs():
        for i in sites:
    #         print(i)
            right = (i + 1) % L 
            # ignore wraparound coordinates if not cyclic
            if cyclic or right != 0:
                yield (i, right)

    # generate all pairs of coordinates and directions
    pairs_ss = tuple(gen_pairs())
    # pairs_ss = tuple(itertools.product(gen_pairs(), 'xyz'))
    print(pairs_ss)
#     print(sites)

    # make XX, YY and ZZ interaction from pair_s
    #     e.g. arg ([(3, 4), (3, 5)], 'z')
    def interactions(pair):
        Sz = spin_operator('z', sparse=True)
        return -ikron([2*Sz, 2*Sz], dims, inds=pair)

    # function to make Z field at ``site``
    def fields(site):
        Sx = spin_operator('x', sparse=True)
        Sz = spin_operator('z', sparse=True)
        return -ikron(g * 2*Sx+ h * 2*Sz, dims, inds=[site])

    # combine all terms
    all_terms = itertools.chain(map(interactions, pairs_ss),
                                map(fields, sites))
    H = sum(all_terms)

    # can improve speed of e.g. eigensolving if known to be real
    if isreal(H):
        H = H.real

    if not sparse:
        H = qarray(H.A)
    else:
        H= quimb.core.sparse_matrix(H)

    return H

# +
α = 10 # Renyi index
La = 4
Lb = La
Lab = La + Lb
Lc_sw = np.arange(2,16)

Nrep = 100
# -

r = 2
Nab = r**Lab
dims = [r] * Lab

beta_sw = [1/4,1/2,0]

ge0={"8":13.0528867, "10":16.499332, "12":19.94577803, "14":23.392224094, "16":26.83867014, "18":30.28511620}


# +
H1 = ising_chain(L=Lab, g=1.05, h=-0.5, cyclic=False,sparse=True)

for i_c in range(len(Lc_sw)):
    Lc = Lc_sw[i_c]    
    Nc = r**Lc


    t_timer=time.time()
    for i_beta in range(len(beta_sw)):
        beta0 = beta_sw[i_beta]

        print("beta is ", beta0)
        beta = beta0/(ge0["%d" % (Lab)]/Lab)
        Nbeta = 6
   
        Rn = np.zeros(Nrep)
        RTn = np.zeros(Nrep)
        for i_r in range(Nrep):
            if i_r%10 ==0:
                print("(Labc, r): ", La, Lb, Lc, i_r)
            #### no symmetry

            X=np.random.randn(Nab,Nc)+1j*np.random.randn(Nab,Nc)
    
            if beta >0:
                for i_b in range(Nbeta):
                    X -= (beta/Nbeta/2)*dot(H1,X)

            psi = normalize(np.reshape(X,[Nab*Nc,1]))
            if Lab<Lc:
                rho = lazy_ptr_linop(psi, dims=[r]*(Lab+Lc), sysa=np.arange(Lab))
            else:
                rho = lazy_ptr_linop(psi, dims=[r]*(Lab+Lc), sysa=np.arange(Lab,Lab+Lc))
            rT = lazy_ptr_ppt_linop(psi, dims=[r]*(Lab+Lc), sysa=np.arange(La), sysb=np.arange(La,Lab))
            RTn[i_r] = -log(approx_spectral_function(rT, lambda x: x**α))
            Rn[i_r] = -log(approx_spectral_function(rho, lambda x: x**α))    
        
        f1= 'R%d_b_%.2f_Labc_%d_%d_%d.npz' % (α,beta0,La,Lb,Lc)
        out_dir = 'thermal_data/' 
        fname = out_dir+f1
        np.savez(fname, Rn=Rn, RTn=RTn)

    elapsed = time.time() - t_timer
    print("Finished, elapsed time = %.2f " % (elapsed)+ "sec")
