# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2024, The University of Texas at Austin 
# & University of California--Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import ufl

from ..algorithms.linSolvers import PETScKrylovSolver
from ..utils.vector2function import vector2Function

class TVPrior:
    """
    This class implements the primal-dual formulation for the total variation prior.
    
    """
    # [1] Chan, Tony F., Gene H. Golub, and Pep Mulet. "A nonlinear primal-dual method for total variation-based image restoration." SIAM journal on scientific computing 20.6 (1999): 1964-1977.
    
    # primal-dual implementation for (vector) total variation prior
    def __init__(self, Vhm:dl.FunctionSpace, Vhw:dl.FunctionSpace, Vhwnorm:dl.FunctionSpace, alpha:float, beta:float, peps:float=1e-3, rel_tol:float=1e-12, max_iter:int=100):
        self.alpha = dl.Constant(alpha)
        self.beta = dl.Constant(beta)
        
        self.Vhm = Vhm  # function space for the parameter
        self.Vhw = Vhw  # function space for the slack variable
        self.Vhwnorm = Vhwnorm  # function space for the norm of the slack variable

        # linearization point
        self.m_lin = None
        self.w_lin = None
        
        self.gauss_newton_approx = False  # by default don't use GN approximation to Hessian
        self.peps = peps  # mass matrix perturbation for preconditioner

        # assemble mass matrix for parameter, slack variable, norm of slack variable
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.m_trial, self.m_test, self.M, self.Msolver = self._setupM(self.Vhm)
        self.w_trial, self.w_test, self.Mw, self.Mwsolver = self._setupM(self.Vhw)
        self.wnorm_trial, self.wnorm_test, self.Mwnorm, self.Mwnormsolver = self._setupM(self.Vhwnorm)


    def _setupM(self, Vh:dl.FunctionSpace):
        # helper function to set up mass matrix, solver
        trial = dl.TrialFunction(Vh)
        test = dl.TestFunction(Vh)
        
        # assemble mass matrix from variational form
        varfM = dl.inner(trial, test)*dl.dx
        M = dl.assemble(varfM)
        
        # set up PETSc solver object to apply M^{-1}
        Msolver = PETScKrylovSolver(Vh.mesh().mpi_comm(), "cg", "jacobi")
        Msolver.set_operator(M)
        Msolver.parameters["maximum_iterations"] = self.max_iter
        Msolver.parameters["relative_tolerance"] = self.rel_tol
        Msolver.parameters["error_on_nonconvergence"] = True
        Msolver.parameters["nonzero_initial_guess"] = False
        
        # return objects
        return trial, test, M, Msolver


    def _fTV(self, m:dl.Function)->dl.Function:
        """Computes the TV functional.

        Args:
            m (dl.Function): Function to compute the TV norm of.

        Returns:
            dl.Function: TV norm of m.
        """
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
    
    
    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)
    
    
    def setLinearizationPoint(self, m:dl.Vector, w:dl.Vector, gauss_newton_approx:bool):
        self.m_lin = vector2Function(m, self.Vhm)
        self.w_lin = vector2Function(w, self.Vhw)
        self.gauss_newton_approx = gauss_newton_approx
    
    
    def cost(self, m):
        # (smoothed) TV functional
        m = vector2Function(m, self.Vhm)
        return dl.assemble( self.alpha * self._fTV(m)*dl.dx )
    
    
    def grad(self, m, out):
        out.zero()
        m = vector2Function(m, self.Vhm)
        
        TVm = self._fTV(m)
        grad_tv = self.alpha * dl.Constant(1.)/TVm*dl.inner(dl.grad(m), dl.grad(self.m_test))*dl.dx
        
        # assemble the UFL form to a vector, add to out
        dl.assemble(grad_tv, tensor=out)
    
    
    def hess_action(self, m, w, m_dir):
        TVm = self._fTV(m)
        
        # symmetrized version of (5.2) from [1]
        A = dl.Constant(1.)/TVm * ( dl.Identity(2) 
                                    - dl.Constant(0.5)*dl.outer(w, dl.grad(m)/TVm)
                                    - dl.Constant(0.5)*dl.outer(dl.grad(m)/TVm, w) )
        
        return self.alpha * dl.inner(A*dl.grad(m_dir), dl.grad(self.m_test))*dl.dx
    
    
    def applyR(self, dm, out):
        out.zero()  # zero out the output
        
        m_dir = vector2Function(dm, self.Vhm)
        hessian_action_form = self.hess_action(self.m_lin, self.w_lin, m_dir)
        
        dl.assemble(hessian_action_form, tensor=out)
    
    
    def compute_w_hat(self, m, w, m_hat, w_hat):
        m = vector2Function(m, self.Vhm)
        m_hat = vector2Function(m_hat, self.Vhm)
        w = vector2Function(w, self.Vhw)
        
        TVm = self._fTV(m)
        
        # symmetrized version of operator in (3.6) from [1]
        A = dl.Constant(1.)/TVm * ( dl.Identity(2) 
                                   - dl.Constant(0.5)*dl.outer(w, dl.grad(m)/TVm)
                                   - dl.Constant(0.5)*dl.outer(dl.grad(m)/TVm, w) )
        
        # expression for incremental slack variable (3.6) from [1]
        dw = A*dl.grad(m_hat) - w + dl.grad(m)/TVm
        dw = dl.assemble( dl.inner(self.w_test, dw)*dl.dx )
        
        # project into appropriate space
        self.Mwsolver.solve(w_hat, dw)
    
    
    def wnorm(self, w):
        w = vector2Function(w, self.Vhw)
        
        # compute functional and assemble
        nw = dl.inner(w, w)
        nw = dl.assemble( dl.inner(self.wnorm_test, nw)*dl.dx )
        
        # project into appropriate space
        out = dl.Vector(self.Mwnorm.mpi_comm())
        self.Mwnorm.init_vector(out, 0)
        self.Mwnormsolver.solve(out, nw)
        
        return out
    
    
    def generate_slack(self):
        """ Return a vector in the shape of the slack variable. """
        return dl.Function(self.Vhw).vector()
    
    
    def compute_w(self, m:dl.Vector):
        m = vector2Function(m, self.Vhm)
        
        TVm = self._fTV(m)
        w = dl.grad(m)/TVm
        w = dl.assemble( dl.inner(self.w_test, w)*dl.dx )
        
        out = dl.Vector(self.Mw.mpi_comm())
        self.Mw.init_vector(out, 0)
        self.Mwsolver.solve(out, w)
        return out
    
    
    def Psolver(self):
        # set up the preconditioner for the Hessian
        varfHTV = self.hess_action(self.m_lin, self.w_lin, self.m_trial)
        varfM = dl.inner(self.m_trial, self.m_test)*dl.dx
        varfP = varfHTV + self.peps*varfM
        
        # assemble the preconditioner and set as operator for solver
        P = dl.assemble(varfP)
        Psolver = PETScKrylovSolver(self.Vhm.mesh().mpi_comm(), "cg", "hypre_amg")
        Psolver.set_operator(P)
        Psolver.parameters["nonzero_initial_guess"] = False
        
        return Psolver
    
    
    def mpi_comm(self):
        return self.Vhm.mesh().mpi_comm()
        
    
# class TVGaussianPrior:
#     # primal implementation
#     raise NotImplementedError("Fused TV+Gaussian Prior primal formulation not yet implemented.")
