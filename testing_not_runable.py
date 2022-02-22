#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 16:05:59 2022

@author: sxyang
"""
import numpy as np

# CHSH test for checking points to be local or nonlocal.
# CHSH coeff, Pa0 Pa1 Pb0 Pb1 Pab00 Pab10 Pab01 Pab11
beta = np.asarray([-1, 0, -1, 0, 1, 1, 1, -1])*4

for ext in exts:
    print(sum(beta * ext))
    
# Vrifiying the MUBs from mubs(o) are correct for d = prime.
# <e|f> = 1/d, e, f are basis from different MUBs.
test for d==4.
for i in range(d+1):
    for j in range(d+1):
        if i != j:
            vec1 = mat[i]
            vec2 = mat[j]
            for l in range(d):
                for k in range(d):
                    test_val = abs((vec1[l] @ np.conj(vec2[k].T))**2)
                    if abs(test_val - 1/d) > 0.0001 :
                        print('i,j=', i, j)
                        print(test_val)