#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 13:58:11 2022

@author: sxyang
"""
import numpy as np


def Gen_Local_ExtPts_Bipartite_MO(m, o):
    # Local strategies can be understood as the participants always picks certain outcome, ex. P(a=0,x) = 1, P(a!=0, x) = 0
    # We only consider N = 2 parties here.
    N = 2
    num_local_strategy = o**m
    num_ExtPts = num_local_strategy**2
    # Pax, Pby, Pabxy, num of marginal terms + composition terms.
    dim_Extpts = (o-1)*m*N + ((o-1)*m)**N
    
    Pax = np.zeros((o*m - m, num_local_strategy))
    
    Ext_indx = np.arange(num_local_strategy)
    o_uni = [] 
    for i in range(o):
        tmp = np.zeros(o)
        tmp[i] = 1
        o_uni.append(tmp)
    
    marg = []
    combine = []
    tmp_pt = []
    for num_LS in Ext_indx:
        combine = np.unravel_index(num_LS, [o]*m)
        for ind_m in range(m):
            # marg is obtained from local strategies.
            tmp_pt += list(o_uni[combine[ind_m]])
            # sum_a Pax = 1 >> only need o-1
            tmp_pt.pop(o-1)
        marg.append(tmp_pt.copy())
        tmp_pt = []
    # ------------- Marginal Expts done. -------------
    marg_Exts = np.asarray(marg)
    # Pabxy -> Extpts
    Extpts = np.tensordot(marg_Exts, marg_Exts, axes=0)
    # (num_marg, Pabxy/2, num_marg, Pabxy/2) -> (num_ExtPts, Pabxy)
    Extpts = np.swapaxes(Extpts, 1, 2).reshape(num_ExtPts, -1)
    N_marg = np.zeros((num_ExtPts, (o-1)*m*N))
    for ind_extpts in range(num_ExtPts):
        combine = np.unravel_index(ind_extpts, [num_local_strategy]*N)
        tmp_marg = marg_Exts[combine[0]].reshape(1, -1)
        for ind_m in range(1, m):
            tmp_marg = np.concatenate((tmp_marg, marg_Exts[combine[ind_m]].reshape(1,-1)), axis=1).reshape(1,-1)
            
        N_marg[ind_extpts] = tmp_marg
    Extpts = np.concatenate((N_marg, Extpts), axis=1)
    
    return Extpts
    # --------- another way to get Extpts. -----------
    # Extpts = np.zeros((num_ExtPts, dim_Extpts))
    # for ind_extpts in range(num_ExtPts):
    #     combine = np.unravel_index(ind_extpts, [num_local_strategy]*N)
    #     tmp_ext = marg_Exts[combine[0]].reshape(1, -1)
    #     tmp_marg = marg_Exts[combine[0]].reshape(1, -1)
    #     for ind_m in range(1, m):
    #         tmp_marg = np.concatenate((tmp_marg, marg_Exts[combine[ind_m]].reshape(1,-1)), axis=1).reshape(1,-1)
    #         tmp_ext = np.tensordot(tmp_ext, marg_Exts[combine[ind_m]], axes=0).reshape(1, -1)
            
    #     Extpts[ind_extpts] = np.concatenate((tmp_marg, tmp_ext), axis=1)
        
        #------------------    

