import numpy as np
from quimb import *
import quimb.tensor as qtn
from quimb.tensor import *
import csv
import time
from numpy.linalg import matrix_power

N = 14
bond_dims  = [2,4,8,16,32,64,999]
runs = 100

with open('training_data_N_%d_eff.csv' % (N), mode='a') as training_data:
 
    for i in np.arange(runs):
        t = time.time()
        dims = [2] * N
        D = prod(dims)
        NAs = np.arange(2, N+1)
        for bond_dim in bond_dims:
            if bond_dim == 999:
                
                randket= rand_haar_state(D)
            
            else:
                randket= rand_matrix_product_state(N, bond_dim, phys_dim=2, dtype=complex, cyclic=True, trans_invar=False)
            
            for NA in NAs:
                NB = int(N - NA)
                randstate = ptr(randket, dims, np.arange(NA))
                if NA<N:
                    NA1s = np.arange(1, NA)
                    for NA1 in NA1s:
                        NA2 = int(NA - NA1)

                        LN = logneg_subsys(randket, dims, np.arange(NA1), np.arange(NA1,NA))

                        randstate_pt = partial_transpose(randstate, [2] * NA, sysa=np.arange(NA1))
                        r2 = np.dot(randstate_pt, randstate_pt)
                        r3 = np.dot(r2, randstate_pt)
                        r4 = np.dot(r3, randstate_pt)
                        r5 = np.dot(r4, randstate_pt)
                        r6 = np.dot(r5, randstate_pt)
                        r7 = np.dot(r6, randstate_pt)
                        r8 = np.dot(r7, randstate_pt)

                        p2 = np.real(np.trace(r2))
                        p3 = np.real(np.trace(r3))
                        p4 = np.real(np.trace(r4))
                        p5 = np.real(np.trace(r5))
                        p6 = np.real(np.trace(r6))
                        p7 = np.real(np.trace(r7))
                        p8 = np.real(np.trace(r8))
                        
                        #Writing to file
                        training_writer = csv.writer(training_data, delimiter=',')
                        training_writer.writerow([NA1,NA2,NB,p2,p3,p4,p5,p6,p7,p8,LN])
                else:
                    NA1s = np.arange(1, NA)
                    for NA1 in NA1s:
                        NA2 = int(NA - NA1)

                        LN = logneg_subsys(randket, dims, np.arange(NA1), np.arange(NA1,NA))

                        rA1 = partial_trace(randstate, [2]*NA , keep=np.arange(NA1))
                        r2 = np.dot(rA1, rA1)
                        r3 = np.dot(r2, rA1)
                        r4 = np.dot(r3, rA1)
                        r5 = np.dot(r4, rA1)
                        r6 = np.dot(r5, rA1)
                        r7 = np.dot(r6, rA1)

                        p2 = np.real(np.trace(rA1))
                        p3 = np.real(np.trace(r3))
                        p4 = np.real(np.trace(r2))**2
                        p5 = np.real(np.trace(r5))
                        p6 = np.real(np.trace(r3))**2
                        p7 = np.real(np.trace(r7))
                        p8 = np.real(np.trace(r4))**2
                
                    
                        #Writing to file
                        training_writer = csv.writer(training_data, delimiter=',')
                        training_writer.writerow([NA1,NA2,NB,p2,p3,p4,p5,p6,p7,p8,LN])

        print('Round:',i+1, round(time.time()-t,3), 'seconds')
    
    
print("Done!")

