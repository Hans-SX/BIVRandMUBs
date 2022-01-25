#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:08:22 2022

@author: sxyang
"""

import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import hadamard

omega = np.exp(1j*2*np.pi/4)
X = np.asarray([[0, 1], [1, 0]], dtype=complex)
Z = np.asarray([[1, 0], [0, -1]], dtype=complex)

mat = np.zeros((5,4,4), dtype=complex)

mat[0] = np.eye(4, dtype=complex)
mat[1] = hadamard(4, dtype=complex)/2
mat[2] = (np.kron(Z, 1j*np.ones((2,2), dtype=complex))) + np.kron(X, [[-1, 1], [1, -1]])/2

mat[3] = np.asarray([[1, 1j, 1, -1j], [1j, -1, -1j, -1], [1j, 1, 1j, -1], [-1, 1j, 1, 1j]])/2
mat[4] = np.asarray([[1j, -1, 1j, -1], [-1, -1j, 1, 1j], [1j, 1, 1j, 1], [1, -1j, -1, 1j]])/2

for i in range(5):
    for j in range(5):
        for l in range(4):
            for k in range(4):
                if i != j:
                    print('i,j=', i, j)
                    print(abs((mub4[i][:][l]@mub4[j][:][k].T)**2))
        
for i in range(5):
    for j in range(5):
        if i != j:
            print('i,j=', i, j)
            print(sum(sum(mub4[i]@mub4[j])))
def mubs(o):
    # mubs in correlator form.
    # For prime o is okay, otherwise might not be correct.
    bra_j = np.identity(o, dtype=complex)
    w = np.exp(2*np.pi*1j/o)
    X = np.zeros((o,o), dtype=complex)
    Z = np.zeros((o,o), dtype=complex)
    for j in range(o):
        X += np.tensordot(bra_j[(j+1)%o].T, bra_j[j], axes=0)
        Z += np.tensordot(bra_j[j].T, bra_j[j], axes=0) * w**j
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
            povm[ind_o, ind_povm] = np.tensordot(ZX_o[ind_o][:][ind_povm], ZX_o[ind_o][:][ind_povm].T, axes=0)
    return povm, ZX_o

def ProbDist_2mo(m, o):
    
    mubs_povm = mubs(o)
    
    M = mubs_povm[:m]
    ua = unitary_group.rvs(o)
    ub = unitary_group.rvs(o)
    # ---------- here, reconsider the shape on the following, above is correct -----------------------
    Ma = np.zeros((o-1, m))
    Mb = np.zeros((o-1, m))
    for ind_o in range(o-1):
        for ind_m in rang(m):
            Ma = ua @ M[ind_o, ind_m] @ np.conj(ua.T)
            Mb = ub @ M[ind_o, ind_m] @ np.conj(ub.T)
        
    Pax = np.zeros((o-1, m))
    Pby = np.zeros((o-1, m))
    for ind_o in range(o-1):
        for ind_m in range(m):
            Pax[ind_o, ind_m] = np.trace(rho_a @ Ma[ind_m])