#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:05:28 2022

@author: sxyang
"""
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import hadamard
import cvxpy as cp

from Gen_Local_ExtPts_Bipartite_MO import Gen_Local_ExtPts_Bipartite_MO
from Gen_Local_ExtPts_Bipartite_MO import Gen_Local_ExtPts_Bipartite_MO_jointprob
from ProbDist_2mo import ProbDist_2mo_v1, ProbDist_2mo_vAlljoint, mubs
from vis_2mo_linpro import vis_2mo
from itertools import combinations

def m_from_MUBs(m, num_povm):
      com = combinations(np.arange(num_povm), m)
      m_from_mubs = list(com)
      A_combine_B = []
      for ind in range(len(m_from_mubs)**2):
          A_combine_B.append(np.unravel_index(ind, [len(m_from_mubs)]*2))
      return m_from_mubs, A_combine_B


def vis_2mo_results(m, o, num_pts, seed):
    # exts = Gen_Local_ExtPts_Bipartite_MO(m, o)
    exts = Gen_Local_ExtPts_Bipartite_MO_jointprob
    # num_pax = (o-1)*m
    num_pabxy = ((o-1)*m)**2
    # probability distribution of white noise
    pw = np.ones(num_pabxy)/o**2
    # pw = np.concatenate((np.ones(2*num_pax)/o, np.ones(num_pabxy)/o**2)).reshape(1, -1)
    
    if o == 4:
        mat = np.zeros((5,4,4), dtype=complex)
    
        mat[0] = np.eye(4, dtype=complex)
        mat[1] = hadamard(4, dtype=complex)/2
        mat[2] = np.asarray([[1, -1, -1j, -1j], [1, -1, 1j, 1j], [1, 1, 1j, -1j], [1, 1, -1j, 1j]])/2
        mat[3] = np.asarray([[1, -1j, -1j, -1], [1, -1j, 1j, 1], [1, 1j, 1j, -1], [1, 1j, -1j, 1]])/2
        mat[4] = np.asarray([[1, 1j, 1, -1j], [1, 1j, -1, 1j], [1, -1j, 1, 1j], [1, -1j, -1, -1j]])/2
        mubs_povm = mat
    # elif o == 6:
    #     mubs_povm = 
    else:
        mubs_povm, _ = mubs(o)
    
    m_combine, ab_ind = m_from_MUBs(m, o+1)
    
    solver = 0
    # solver = cp.GLPK_MI
    # solver = 'ECOS_BB'
    vis = np.zeros(num_pts)
    vis_tmp = np.zeros(len(ab_ind))
    np.random.seed(seed)
    for i in range(num_pts):
        ua = unitary_group.rvs(o)
        ub = unitary_group.rvs(o)
        for ind, com in enumerate(ab_ind):
            ma = mubs_povm[list(m_combine[com[0]])]
            mb = mubs_povm[list(m_combine[com[1]])]
    
            pt = ProbDist_2mo_v1(m, o, ma, mb, ua, ub)
            vis_tmp[ind] = vis_2mo(pt, exts, m, o, pw, solver)
        vis[i] = min(vis_tmp)
    return vis
