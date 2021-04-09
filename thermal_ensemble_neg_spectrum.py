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
import quimb

r=2 # local Hilbert space dim

def ising_chain(L, g=1.0, h=0.0, cyclic=True,
                sparse=True):

# g=0
# h=0
# cyclic=True
# m= 4
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

La = 3
Lb = La
Lab = La + Lb
Lc = 12
Nrep = 1

r = 2
Nab = r**Lab
Nc = r**Lc
dims = [r] * Lab

# # chaotic
H1 = ising_chain(L=Lab, g=1.05, h=-0.5, cyclic=False,sparse=True)
# ge0 , _ = scipy.sparse.linalg.eigsh(-H1,1)
ge0 = groundenergy(H1)

print(ge0)

# v1b=np.zeros(Nrep*Nab)

# beta = 1/(4*np.abs(ge0)/Lab)
# Nbeta = 4

# for i_r in range(Nrep):
#     #### no symmetry
#     X=np.random.randn(Nab,Nc)+1j*np.random.randn(Nab,Nc)
        
#     if beta >0:
#         for i_b in range(Nbeta):
#             X -= (beta/Nbeta/2)*dot(H1,X)
#     mat= dot(X,np.matrix(X).H)
#     rho= mat / np.trace(mat)
#     rT = partial_transpose(rho, dims=dims, sysa=np.arange(La))
#     l1T=np.linalg.eigvalsh(rT)
#     v1b[i_r*Nab:(i_r+1)*Nab] = l1T #*(Nab)

# print("done!")
