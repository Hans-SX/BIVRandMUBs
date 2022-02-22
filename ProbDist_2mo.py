#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:08:22 2022

@author: sxyang
"""

import numpy as np


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

def ProbDist_2mo_v1(m, o, ma, mb, ua, ub):
        
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

    