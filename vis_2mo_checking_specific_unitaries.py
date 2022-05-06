#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:05:28 2022

@author: sxyang
"""
# %%
import numpy as np
import scipy
from scipy.linalg import hadamard
from scipy.stats import unitary_group
from scipy.io import savemat
# import cvxpy as cp
import matplotlib.pyplot as plt

from Gen_Local_ExtPts_Bipartite_MO import Gen_Local_ExtPts_Bipartite_MO_jointprob
from ProbDist_2mo import ProbDist_2mo_vAlljoint, mubs, ProbDist_2mo_Alljoint_vkron
from vis_2mo_linpro import vis_2mo
from itertools import combinations

def k_MUBs_choose_m(num_povm, m=2):
      com = combinations(np.arange(num_povm), m)
      m_from_mubs = list(com)
      A_combine_B = []
      for ind in range(len(m_from_mubs)**2):
          A_combine_B.append(np.unravel_index(ind, [len(m_from_mubs)]*2))
      return m_from_mubs, A_combine_B

def vis_2mo_spec_u(m, k, o):
    # exts = Gen_Local_ExtPts_Bipartite_MO(m, o)
    exts = Gen_Local_ExtPts_Bipartite_MO_jointprob(m, o)
    # num_pax = (o-1)*m
    num_pabxy = (o*m)**2
    # probability distribution of white noise
    pw = np.ones(num_pabxy)/o**2
    pw = np.reshape(pw, (1,-1))
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

    k2_combine, ab_ind = k_MUBs_choose_m(k)
    povm = mubs_povm[:k]

    solver = 0
    # solver = cp.GLPK_MI
    # solver = 'ECOS_BB'

    ua = np.identity(o, dtype=complex)
    with open('vtest.txt', 'r') as f:
        tmp = f.readlines()    
    f.close()

    lines = []
    for line in tmp:
        lines.append(line.split())
    lines = [ele for ele in lines if ele != []]
    lines = [e2 for e1 in lines for e2 in e1]
    ub_all = np.array(list(map(complex, lines)))
    ub_all = ub_all.reshape(9,3,3)

    vis = np.zeros(ub_all.shape[0])
    vis_tmp = np.zeros(len(ab_ind))
    for i in range(ub_all.shape[0]):
        ub = ub_all[i]
        for ind, com in enumerate(ab_ind):
            ma = povm[list(k2_combine[com[0]])]
            mb = povm[list(k2_combine[com[1]])]
            
            # pt = ProbDist_2mo_vAlljoint(m, o, ma, mb, ua, ub)
            pt = ProbDist_2mo_Alljoint_vkron(m, o, ma, mb, ua, ub)
            # pt = np.reshape(pt, (-1,))
            pt = np.reshape(pt, (1,-1))
            # pt = ProbDist_2mo_v1(m, o, ma, mb, ua, ub)
            vis_tmp[ind] = vis_2mo(pt, exts, m, o, pw, solver)
        vis[i] = min(vis_tmp)
    print(vis)
    print(max(vis), min(vis), np.mean(vis))
    print('k = ', k, ' d = ', o)
    n, bins, patches = plt.hist(vis,40)

    plt.xlabel('visibility')
    plt.ylabel('Relative frequency')
    plt.title('Vis_'+str(k)+str(o))
    plt.show()
    return vis

def vis_2mo_spec_u_v2(ub_all, k=4, o=3):
    m = 2
    # exts = Gen_Local_ExtPts_Bipartite_MO(m, o)
    exts = Gen_Local_ExtPts_Bipartite_MO_jointprob(m, o)
    # num_pax = (o-1)*m
    num_pabxy = (o*m)**2
    # probability distribution of white noise
    pw = np.ones(num_pabxy)/o**2
    pw = np.reshape(pw, (1,-1))
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

    k2_combine, ab_ind = k_MUBs_choose_m(k)
    povm = mubs_povm[:k]

    solver = 0
    # solver = cp.GLPK_MI
    # solver = 'ECOS_BB'

    ua = np.identity(o, dtype=complex)

    vis = np.zeros(ub_all.shape[0])
    vis_tmp = np.zeros(len(ab_ind))
    for i in range(ub_all.shape[0]):
        ub = ub_all[i]
        for ind, com in enumerate(ab_ind):
            ma = povm[list(k2_combine[com[0]])]
            mb = povm[list(k2_combine[com[1]])]
            
            # pt = ProbDist_2mo_vAlljoint(m, o, ma, mb, ua, ub)
            pt = ProbDist_2mo_Alljoint_vkron(m, o, ma, mb, ua, ub)
            # pt = np.reshape(pt, (-1,))
            pt = np.reshape(pt, (1,-1))
            # pt = ProbDist_2mo_v1(m, o, ma, mb, ua, ub)
            vis_tmp[ind] = vis_2mo(pt, exts, m, o, pw, solver)
        vis[i] = min(vis_tmp)
    # print(vis)
    print(max(vis), min(vis), np.mean(vis))
    print('k = ', k, ' d = ', o)
    n, bins, patches = plt.hist(vis,40)

    plt.xlabel('visibility')
    plt.ylabel('Relative frequency')
    plt.title('Vis_'+str(k)+str(o))
    plt.show()
    return vis

def gen_unitary_family(unitary, t, num):
    d = unitary.shape[0]
    W = np.zeros((num, d, d), dtype=complex)
    for i in range(num):
        tmp = unitary_group.rvs(d)
        H = tmp + np.conj(tmp.T)
        V = scipy.linalg.expm(1j*t*H)
        W[i] = unitary @ V
    return W

#%%
if __name__ == "__main__":
    k = 4; d = 3 # fixed for verifying
    # vis = vis_2mo_spec_u(2,k,d)
#     print(vis[vis>=1])
#     print(sum(vis<1))
#     print(max(vis), min(vis), np.mean(vis))
#     print('k = ', k, ' d = ', d)    
#     # print(vis222)

    with open('vtest.txt', 'r') as f:
        tmp = f.readlines()    
    f.close()

    lines = []
    for line in tmp:
        lines.append(line.split())
    lines = [ele for ele in lines if ele != []]
    lines = [e2 for e1 in lines for e2 in e1]
    ub_all = np.array(list(map(complex, lines)))
    ub_all = ub_all.reshape(9,3,3)

    t = 0.0001
    num = 10000
    ub_fam = gen_unitary_family(ub_all[1], t, num)
    
    # vis = vis_2mo_spec_u_v2(ub_fam, k=4, o=3)
    vis = vis_2mo_spec_u_v2(ub_all, k=4, o=3)
    # np.savetxt('ub_family', ub_fam.reshape(-1,3))
    # savemat('ub_family.mat', {'UV': ub_fam[:4]})
# %%
