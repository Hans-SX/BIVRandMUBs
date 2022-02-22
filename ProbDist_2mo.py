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
# <e|f> = 1/d, e, f are basis from different MUBs.
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

def ProbDist_2mo_v1(m, o, ma, mb, ua, ub):
    
    '''
    Takde out this part.
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
        
        # From Gelo's code, seems not correct, above from wiki is correct.
        # mat[0] = np.eye(4, dtype=complex)
        # mat[1] = hadamard(4, dtype=complex)/2
        # mat[2] = ( np.kron(Z, 1j*np.ones((2,2))) + np.kron(X, [[-1, 1], [1, -1]]) )/2
        # mat[3] = np.asarray([[1, 1j, 1, -1j], [1j, -1, -1j, -1], [1j, 1, 1j, -1], [-1, 1j, 1, 1j]])/2
        # mat[4] = np.asarray([[1j, -1, 1j, -1], [-1, -1j, 1, 1j], [1j, 1, 1j, 1], [1, -1j, -1, 1j]])/2
        # mubs_povm = mat
    # elif o == 6:
    #     mubs_povm = 
    else:
        mubs_povm, _ = mubs(o)
        '''
        
    kets = []
    state = np.zeros(o**2, dtype=complex)
    for d in range(o):
        ket = np.zeros(o, dtype=complex)
        ket[d] = 1
        kets.append(ket)
        state += np.kron(kets[d], kets[d])
    state = state.reshape(o**2, 1) / np.sqrt(o)
    rho = np.kron(state, np.conj(state.T))
    rho_a = np.trace(rho.reshape(o,o,o,o), axis1=0, axis2=2)
    rho_b = np.trace(rho.reshape(o,o,o,o), axis1=1, axis2=3)
    
    # o-1 for using the constraint sum_a Pax = 1.
    # Pax (a -> ind_o, x -> ind_m), ind_o-th POVM element and ind_m-th measurement.
    Ma = np.zeros((m, o-1, o, o), dtype=complex)
    Mb = np.zeros((m, o-1, o, o), dtype=complex)
    
    num_ele = ((o-1)*m)**2 + (o-1)*m*2
    ProbDist = np.zeros(num_ele)

    for ind_o in range(o-1):
        for ind_m in range(m):
            Ma[ind_m, ind_o] = ua @ ma[ind_m, ind_o] @ np.conj(ua.T)
            Mb[ind_m, ind_o] = ub @ mb[ind_m, ind_o] @ np.conj(ub.T)
        
    Pax = np.zeros((m, o-1), dtype=complex)
    Pby = np.zeros((m, o-1), dtype=complex)
    for ind_o in range(o-1):
        for ind_m in range(m):
            Pax[ind_m, ind_o] = np.trace(rho_a @ Ma[ind_m, ind_o])
            Pby[ind_m, ind_o] = np.trace(rho_b @ Mb[ind_m, ind_o])
            
    Pabxy = np.zeros((m, o-1, m, o-1), dtype=complex)
    for ai in range(o-1):
        for bi in range(o-1):
            for xi in range(m):
                for yi in range(m):
                    Mab = np.tensordot(Ma[xi, ai], Mb[yi, bi], axes=0)
                    # (|a>,<a|,|b>,<b| ) -> (|ab>,<ab|)
                    Mab = np.swapaxes(Mab, 1, 2).reshape(o**2, o**2)
                    Pabxy[xi, ai, yi, bi] = np.trace(rho @ Mab)
    ProbDist = np.concatenate((Pax.reshape(1,-1), Pby.reshape(1, -1), Pabxy.reshape(1,-1)), axis=1).real
    return ProbDist


def ProbDist_2mo(m, o, num_pts):
    
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
        
        # From Gelo's code, seems not correct, above from wiki is correct.
        # mat[0] = np.eye(4, dtype=complex)
        # mat[1] = hadamard(4, dtype=complex)/2
        # mat[2] = ( np.kron(Z, 1j*np.ones((2,2))) + np.kron(X, [[-1, 1], [1, -1]]) )/2
        # mat[3] = np.asarray([[1, 1j, 1, -1j], [1j, -1, -1j, -1], [1j, 1, 1j, -1], [-1, 1j, 1, 1j]])/2
        # mat[4] = np.asarray([[1j, -1, 1j, -1], [-1, -1j, 1, 1j], [1j, 1, 1j, 1], [1, -1j, -1, 1j]])/2
        # mubs_povm = mat
    # elif o == 6:
    #     mubs_povm = 
    else:
        mubs_povm, _ = mubs(o)
        
    kets = []
    state = np.zeros(o**2, dtype=complex)
    for d in range(o):
        ket = np.zeros(o, dtype=complex)
        ket[d] = 1
        kets.append(ket)
        state += np.kron(kets[d], kets[d])
    state = state.reshape(o**2, 1) / np.sqrt(o)
    rho = np.kron(state, np.conj(state.T))
    rho_a = np.trace(rho.reshape(o,o,o,o), axis1=0, axis2=2)
    rho_b = np.trace(rho.reshape(o,o,o,o), axis1=1, axis2=3)
    
    M = mubs_povm[:m]
    # o-1 for using the constraint sum_a Pax = 1.
    # Pax (a -> ind_o, x -> ind_m), ind_o-th POVM element and ind_m-th measurement.
    Ma = np.zeros((m, o-1, o, o), dtype=complex)
    Mb = np.zeros((m, o-1, o, o), dtype=complex)
    
    num_ele = ((o-1)*m)**2 + (o-1)*m*2
    ProbDist = np.zeros((num_pts, num_ele))
    for i in range(num_pts):
        ua = unitary_group.rvs(o)
        ub = unitary_group.rvs(o)
        for ind_o in range(o-1):
            for ind_m in range(m):
                Ma[ind_m, ind_o] = ua @ M[ind_o, ind_m] @ np.conj(ua.T)
                Mb[ind_m, ind_o] = ub @ M[ind_o, ind_m] @ np.conj(ub.T)
            
        Pax = np.zeros((m, o-1), dtype=complex)
        Pby = np.zeros((m, o-1), dtype=complex)
        for ind_o in range(o-1):
            for ind_m in range(m):
                Pax[ind_m, ind_o] = np.trace(rho_a @ Ma[ind_m, ind_o])
                Pby[ind_m, ind_o] = np.trace(rho_b @ Mb[ind_m, ind_o])
                
        Pabxy = np.zeros((m, o-1, m, o-1), dtype=complex)
        for ai in range(o-1):
            for bi in range(o-1):
                for xi in range(m):
                    for yi in range(m):
                        Mab = np.tensordot(Ma[xi, ai], Mb[yi, bi], axes=0)
                        # (|a>,<a|,|b>,<b| ) -> (|ab>,<ab|)
                        Mab = np.swapaxes(Mab, 1, 2).reshape(o**2, o**2)
                        Pabxy[xi, ai, yi, bi] = np.trace(rho @ Mab)
        ProbDist[i] = np.concatenate((Pax.reshape(1,-1), Pby.reshape(1, -1), Pabxy.reshape(1,-1)), axis=1).real
    return ProbDist
    
    
    
    
    