#!/usr/local/bin/python3

import cvxpy as cvx
import numpy as np

from lipschitzexp import LipschitzExp


r = np.array([1,1,1])
x = -r

n = len(r)
p = 1

u = lambda r: LipschitzExp(r,1,0)
q = cvx.Variable(p)


obj = cvx.Minimize(-1/n * cvx.sum_entries(u( cvx.mul_elemwise(r,x*q)  )) + 10*cvx.norm(q)**2)
pr = cvx.Problem(obj)
print('r=',r)
print('x=',x)
val = pr.solve()

print(val)
