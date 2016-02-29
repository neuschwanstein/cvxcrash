import cvxpy as cvx
import numpy as np

from lipschitzexp import LipschitzExp


p = 3
n = 10000

u = lambda r: LipschitzExp(r,1,0)
q = cvx.Variable(p)

r = np.random.normal(8,10,size=n)
x = np.random.multivariate_normal(np.zeros(p),np.eye(p),size=n)

obj = cvx.Minimize(-1/n * cvx.sum_entries(u( cvx.mul_elemwise(r,x*q)  )) + 1500*cvx.norm(q)**2)
pr = cvx.Problem(obj)
val = pr.solve()

print(val)
