#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:17:52 2022

@author: sxyang
"""

import cvxpy as cp

def vis_2mo(pt, exts, m, o, pw, solver=0):
  '''
  pt: a point of probability distribution from entangled state, has the form:
      ({Pax}, {Pby}, {Pabxy})_(a, b = 1 ~ o-1 ).
  exts: local extreme points, the form is the same as pt.
  m: number of measurements
  o: number of outcomes per measurement.
  '''
  # visibility
  v = cp.Variable()
  # coefficients for each extreme point.
  coeff = cp.Variable((1, (o**m)**2))
  # num_pax = (o-1)*m
  # num_pabxy = ((o-1)*m)**2
  # # probability distribution of white noise
  # pw = np.concatenate((np.ones(2*num_pax)/o, np.ones(num_pabxy)/o**2)).reshape(1, -1)
  # print(coeff.shape)
  P = v*pt + (1-v)*pw
  constraints = []
  constraints = constraints + [v>=0]
  # constraints = constraints + [v<=1]
  constraints = constraints + [coeff >= 0]
  constraints = constraints + [cp.sum(coeff, axis=1) == 1]
  constraints = constraints + [coeff @ exts == P]
  obj = cp.Minimize(v)
  prob = cp.Problem(obj, constraints)

  if solver == 0:
     prob.solve()
  else:
     prob.solve(solver=solver)

  # print("status:", prob.status)
  # print("optimal value", prob.value)
  # print("optimal vis", v.value)
  # print("coeff[0]", coeff[0].value)
  return v.value
