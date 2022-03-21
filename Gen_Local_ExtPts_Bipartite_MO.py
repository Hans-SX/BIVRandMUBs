#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 13:58:11 2022

@author: sxyang
"""
import numpy as np

def Gen_Local_ExtPts_Bipartite_MO_jointprob(m, o):
    # Local strategies can be understood as the participants always picks certain outcome, ex. P(a=0,x) = 1, P(a!=0, x) = 0
    # We only consider N = 2 parties here.
    N = 2
    num_local_strategy = o**m
    num_ExtPts = num_local_strategy**2
    # Pax, Pby, Pabxy, num of marginal terms + composition terms.
    # dim_Extpts = (o-1)*m*N + ((o-1)*m)**N
    
    # Pax = np.zeros((o*m - m, num_local_strategy))
    
    Ext_indx = np.arange(num_local_strategy)
    o_uni = []
    for i in range(o):
        tmp = np.zeros(o)
        tmp[i] = 1
        # o_uni:all the possible local strategies.
        o_uni.append(tmp)
    
    marg = []
    combine = []
    tmp_pt = []
    for num_LS in Ext_indx:
        combine = np.unravel_index(num_LS, [o]*m)
        for ind_m in range(m):
            # marg: Pax, is obtained from combination of local strategies from different m.
            # Ordering of marg: P00, P10, P20, ..., P01, ...
            tmp_pt += list(o_uni[combine[ind_m]])
            # sum_a Pax = 1 >> only need o-1
            # tmp_pt.pop(-1)
        marg.append(tmp_pt.copy())
        tmp_pt = []
    # ------------- Marginal Expts done. -------------
    marg_Exts = np.asarray(marg)
    Extpts = np.kron(marg_Exts, marg_Exts)
    '''
    # reshape all Pax -> (num, x, a)
    marg_Exts = marg_Exts.reshape(num_local_strategy, m, o)
    # Pabxy -> Extpts
    Extpts = np.tensordot(marg_Exts, marg_Exts, axes=0)
    # (num_marg, x, a, num_marg, y, b) -> (num_ExtPts, x, a, y, b)
    Extpts = np.moveaxis(Extpts, 3, 1).reshape(num_ExtPts, m, o, m, o)
    # Split to 2 step to emphasize it is form (x,a,y,b) to a column.
    Extpts = Extpts.reshape(num_ExtPts, -1)
    # N_marg: (num, A/B, x/y, a/b)
    '''
    return Extpts

# def Gen_Local_ExtPts_Bipartite_MO(m, o):
#     # Local strategies can be understood as the participants always picks certain outcome, ex. P(a=0,x) = 1, P(a!=0, x) = 0
#     # We only consider N = 2 parties here.
#     N = 2
#     num_local_strategy = o**m
#     num_ExtPts = num_local_strategy**2
#     # Pax, Pby, Pabxy, num of marginal terms + composition terms.
#     # dim_Extpts = (o-1)*m*N + ((o-1)*m)**N
    
#     # Pax = np.zeros((o*m - m, num_local_strategy))
    
#     Ext_indx = np.arange(num_local_strategy)
#     o_uni = []
#     for i in range(o):
#         tmp = np.zeros(o)
#         tmp[i] = 1
#         # o_uni:all the possible local strategies.
#         o_uni.append(tmp)
    
#     marg = []
#     combine = []
#     tmp_pt = []
#     for num_LS in Ext_indx:
#         combine = np.unravel_index(num_LS, [o]*m)
#         for ind_m in range(m):
#             # marg: Pax, is obtained from combination of local strategies from different m.
#             # Ordering of marg: P00, P10, P20, ..., P01, ...
#             tmp_pt += list(o_uni[combine[ind_m]])
#             # sum_a Pax = 1 >> only need o-1
#             tmp_pt.pop(-1)
#         marg.append(tmp_pt.copy())
#         tmp_pt = []
#     # ------------- Marginal Expts done. -------------
#     marg_Exts = np.asarray(marg)
#     # reshape all Pax -> (num, x, a)
#     marg_Exts = marg_Exts.reshape(num_local_strategy, m, o-1)
#     # Pabxy -> Extpts
#     Extpts = np.tensordot(marg_Exts, marg_Exts, axes=0)
#     # (num_marg, x, a, num_marg, y, b) -> (num_ExtPts, x, a, y, b)
#     Extpts = np.moveaxis(Extpts, 3, 1).reshape(num_ExtPts, m, o-1, m, o-1)
#     # Split to 2 step to emphasize it is form (x,a,y,b) to a column.
#     Extpts = Extpts.reshape(num_ExtPts, -1)
#     # N_marg: (num, A/B, x/y, a/b)
#     N_marg = np.zeros((num_ExtPts, N*m*(o-1)))
#     # test_AB = np.zeros(Extpts.shape)
#     for ind_extpts in range(num_ExtPts):
#         # Combine Pax and Pby.
#         combine = np.unravel_index(ind_extpts, [num_local_strategy]*N)
#         # Pax
#         tmp_marg_A = marg_Exts[combine[0]].reshape(1, -1)
#         tmp_marg_B = marg_Exts[combine[1]].reshape(1, -1)
#         tmp_marg = np.concatenate((tmp_marg_A, tmp_marg_B), axis=1).reshape(1,-1)    
#         N_marg[ind_extpts] = tmp_marg
        
#         # tmp_AB = np.tensordot(tmp_marg_A, tmp_marg_B, axes=0).reshape(1,-1)
#         # test_AB[ind_extpts] = tmp_AB
#     # test_ext = np.concatenate((N_marg, test_AB), axis=1)
#     Extpts = np.concatenate((N_marg, Extpts), axis=1)
#     # print(test_ext-Extpts)
#     return Extpts
    
if __name__ == "__main__":
    exts, exts_kron = Gen_Local_ExtPts_Bipartite_MO_jointprob(3,3)
    print(exts - exts_kron)