import numpy as np

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.exp import exp


class LipschitzExp(Elementwise):
    def __init__(self,x,beta=1,x0=0):
        self.beta = self.cast_to_const(beta)
        self.x0 = self.cast_to_const(x0)
        super().__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self,values):
        beta = self.beta.value
        x0 = self.x0.value
        x = values[0]
        exp = np.exp
        return (x >= x0)* (1/beta * (1 - np.exp(-beta*x))) + \
               (x < x0) * (x*np.exp(-beta*x0) + 1/beta*(1-(1+beta*x0)*np.exp(-beta*x0)))

    def sign_from_args(self):
        return u.Sign.UNKNOWN

    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.monotonicity.DECREASING]

    def get_data(self):
        return [self.beta,self.x0]

    def validate_arguments(self):
        '''Check that beta>0.'''
        if not (self.beta.is_positive() and
                self.beta.is_constant() and self.beta.is_scalar() and
                self.x0.is_constant() and self.x0.is_scalar()):
            raise ValueError('beta must be a non-negative scalar constant, and x0 must a constant scalar.')

    @staticmethod
    def graph_implementation(arg_objs,size,data=None):
        x = arg_objs[0]
        beta,x0 = data[0],data[1]
        beta_val,x0_val = beta.value,x0.value

        if isinstance(beta,Parameter):
            beta = lu.create_param(beta,(1,1))
        else:
            beta = lu.create_const(beta.value,(1,1))
        if isinstance(x0,Parameter):
            x0 = lu.create_param(x0,(1,1))
        else:
            x0 = lu.create_const(x0.value,(1,1))

        xi,psi = lu.create_var(size),lu.create_var(size)
        one = lu.create_const(1,(1,1))
        one_over_beta = lu.create_const(1/beta_val,(1,1))
        k = np.exp(-beta_val*x0_val)
        k = lu.create_const(k,(1,1))

        # 1/beta * (1 - exp(-beta*(xi+x0)))
        xi_plus_x0 = lu.sum_expr([xi,x0])
        minus_beta_times_xi_plus_x0  = lu.neg_expr(lu.mul_expr(beta,xi_plus_x0,size))
        exp_xi,constr_exp = exp.graph_implementation([minus_beta_times_xi_plus_x0],size)
        minus_exp_minus_etc = lu.neg_expr(exp_xi)
        left_branch = lu.mul_expr(one_over_beta, lu.sum_expr([one,minus_exp_minus_etc]),size)

        # psi*exp(-beta*r0)
        right_branch = lu.mul_expr(k,psi,size)

        obj = lu.sum_expr([left_branch,right_branch])

        #x-x0 == xi + psi, xi >= 0, psi <= 0
        zero = lu.create_const(0,size)
        constraints = constr_exp
        prom_x0 = lu.promote(x0, size)
        constraints.append(lu.create_eq(x,lu.sum_expr([prom_x0,xi,psi])))
        constraints.append(lu.create_geq(xi,zero))
        constraints.append(lu.create_leq(psi,zero))

        return (obj, constraints)
