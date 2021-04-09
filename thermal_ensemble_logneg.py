# +
import numpy as np
import scipy
from math import pi, sqrt, tanh
import matplotlib.pyplot as plt
import time
from os import path

import itertools
from operator import add
from quimb import *

r=2 # local Hilbert space dim


def ising_chain_partial(La,L, g=1.0, h=0.0, cyclic=True,
                sparse=True):

    dims = [r] * L  # shape (n, m)

    # generate tuple of all site coordinates
    # sites = tuple(itertools.product(range(n)))
    sites= tuple(range(La))

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
#     print(pairs_ss)

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
        H= core.sparse_matrix(H)

    return H

La = 4
Lb = 4
Lab = La + Lb
Lc_sw = np.arange(9,16)
Nrep = 20

r = 2
Nab = r**Lab

beta_sw = [1/2,1/4]

ge0={"8":13.0528867, "10":16.499332, "12":19.94577803, "14":23.392224094, "16":26.83867014, "18":30.28511620}


for i_c in range(len(Lc_sw)):
    Lc = Lc_sw[i_c]    
    Nc = r**Lc

    dims = [r] * (Lab+Lc)

    t_timer=time.time()

    for i_beta in range(len(beta_sw)):
        beta0 = beta_sw[i_beta]

        print("beta is ", beta0)
        beta = beta0/(ge0["%d" % (Lab)]/Lab)
        Nbeta = 6
   
        if beta>0:
            H1 = ising_chain_partial(La=Lab, L=Lab+Lc, g=1.05, h=-0.5, cyclic=False,sparse=True)

        logneg = np.zeros(Nrep)
        for i_r in range(Nrep):
            if i_r%10 ==0:
                print("(Labc, r): ", La, Lb, Lc, i_r)
            #### no symmetry
            X=rand_ket(Nab*Nc)

            if beta >0:
                for i_b in range(Nbeta):
                    X -= (beta/Nbeta/2)*dot(H1,X)
                X = normalize(X)
            logneg[i_r]=logneg_subsys_approx(X, dims=dims, sysa=np.arange(La), sysb=np.arange(La,Lab))

        f1= 'TPS_b_%.2f_Labc_%d_%d_%d.npz' % (beta0,La,Lb,Lc)
        out_dir = 'thermal_data/' 
        fname = out_dir+f1
        np.savez(fname, logneg=logneg)

    elapsed = time.time() - t_timer
    print("Finished, elapsed time = %.2f " % (elapsed)+ "sec")
# -


