#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:08:22 2022

@author: sxyang
"""

import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import hadamard


# d = 4, 6 are not applied the function below, since the formula is for prime dimensions.
def mubs(o):
    # mubs in correlator form and povm.
    # For prime o is okay, otherwise might not be correct.
    ket_j = np.identity(o, dtype=complex)
    w = np.exp(2*np.pi*1j/o)
    X = np.zeros((o,o), dtype=complex)
    Z = np.zeros((o,o), dtype=complex)
    for j in range(o):
        X += np.tensordot(ket_j[:,(j+1)%o], np.conj(ket_j[:,j].T), axes=0)
        Z += np.tensordot(ket_j[:,j], np.conj(ket_j[:,j].T), axes=0) * w**j
    ZX_o = np.zeros((o+1, o, o), dtype=complex)
    ZX_o[0] = Z
    ZX_o[1] = X
    for ind_o in range(2, o+1):
        X_o = X
        for num_X in range(ind_o-2):
            X_o = X_o @ X
        ZX_o[ind_o] = Z @ X_o
        
    povm = np.zeros((o+1, o, o, o), dtype=complex)
    for ind_o in range(o+1):
        for ind_povm in range(o):
            _, vec = np.linalg.eig(ZX_o[ind_o])
            povm[ind_o, ind_povm] = np.tensordot(vec[ind_povm], np.conj(vec[ind_povm]).T, axes=0)
    return povm, ZX_o

# Vrifiying the MUBs from mubs(o) are correct for d = prime.
d = 5
_, test_mub = mubs(d)
for i in range(d+1):
    for j in range(d+1):
        if i != j:
            _, vec1 = np.linalg.eig(test_mub[i])
            _, vec2 = np.linalg.eig(test_mub[j])
            for l in range(d):
                for k in range(d):
                    test_val = abs((vec1[:,l] @ np.conj(vec2[:,k].T))**2)
                    if abs(test_val - 1/d) > 0.0001 :
                        print('i,j=', i, j)
                        print(test_val)


def ProbDist_2mo(m, o):
    
    if o == 4:
        omega = np.exp(1j*2*np.pi/4)
        X = np.asarray([[0, 1], [1, 0]], dtype=complex)
        Z = np.asarray([[1, 0], [0, -1]], dtype=complex)

        mat = np.zeros((5,4,4), dtype=complex)

        mat[0] = np.eye(4, dtype=complex)
        mat[1] = hadamard(4, dtype=complex)/2
        mat[2] = np.asarray([[1, -1, -1j, -1j], [1, -1, 1j, 1j], [1, 1, 1j, -1j], [1, 1, -1j, 1j]])/2
        mat[3] = np.asarray([[1, -1j, -1j, -1], [1, -1j, 1j, 1], [1, 1j, 1j, -1], [1, 1j, -1j, 1]])/2
        mat[4] = np.asarray([[1, 1j, 1, -1j], [1, 1j, -1, 1j], [1, -1j, 1, 1j], [1, -1j, -1, -1j]])/2
        mubs_povm = mat
    elif o == 6:
        mubs_povm = 
    else:
        mubs_povm, _ = mubs(o)
        
    ket_0 = np.array([1,0], dtype=complex).reshape(2,1)
    ket_1 = np.array([0,1], dtype=complex).reshape(2,1)
    state = (np.kron(ket_0, ket_0) + np.kron(ket_1, ket_1))/np.sqrt(2)
    rho = np.kron(state, np.conj(state.T))
    rho_a = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)
    rho_b = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
    
    M = mubs_povm[:m]
    ua = unitary_group.rvs(o)
    ub = unitary_group.rvs(o)
    # ---------- here, reconsider the shape on the following, above is correct -----------------------
    # o-1 for using the constraint sum_a Pax = 1.
    # Pax (a -> ind_o, x -> ind_m)
    Ma = np.zeros((o-1, m, o, o))
    Mb = np.zeros((o-1, m, o, o))
    for ind_o in range(o-1):
        for ind_m in rang(m):
            Ma[ind_o, ind_m] = ua @ M[ind_o, ind_m] @ np.conj(ua.T)
            Mb[ind_o, ind_m] = ub @ M[ind_o, ind_m] @ np.conj(ub.T)
        
    Pax = np.zeros((o-1, m))
    Pby = np.zeros((o-1, m))
    for ind_o in range(o-1):
        for ind_m in range(m):
            Pax[ind_o, ind_m] = np.trace(rho_a @ Ma[ind_m, ind_o])
            Pby[ind_o, ind_m] = np.trace(rho_b @ Mb[ind_m, ind_o])
            
    