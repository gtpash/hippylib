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
import numpy as np

from ..algorithms.linSolvers import PETScKrylovSolver

class TVPrior:
    
    # [1] Chan, Tony F., Gene H. Golub, and Pep Mulet. "A nonlinear primal-dual method for total variation-based image restoration." SIAM journal on scientific computing 20.6 (1999): 1964-1977.
    
    # primal-dual implementation for (vector) total variation prior
    def __init__(self, Vhm:dl.FunctionSpace, Vhw:dl.FunctionSpace, Vhwnorm:dl.FunctionSpace, alpha:float, beta:float, rel_tol:float=1e-12, max_iter:int=100):
        self.alpha = dl.Constant(alpha)
        self.beta = dl.Constant(beta)
        
        self.Vhm = Vhm  # function space for the parameter
        self.Vhw = Vhw  # function space for the slack variable
        self.Vwnorm = Vhwnorm  # function space for the norm of the slack variable

        # assemble mass matrix for parameter
        self.m_hat = dl.TrialFunction(Vhm)
        self.m_tilde  = dl.TestFunction(Vhm)
        
        varfM = ufl.inner(self.m_hat, self.m_tilde)*ufl.dx
        self.M = dl.assemble(varfM)
        self.Msolver = PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False
        
        # assemble mass matrix for slack variable
        self.w_hat = dl.TrialFunction(Vhw)
        self.w_tilde  = dl.TestFunction(Vhw)
        
        varfMw = dl.inner(self.w_hat, self.w_tilde)*dl.dx
        self.Mw = dl.assemble(varfMw)
        self.Mwsolver = PETScKrylovSolver(self.Vhw.mesh().mpi_comm(), "cg", "jacobi")
        self.Mwsolver.set_operator(self.Mw)
        self.Mwsolver.parameters["maximum_iterations"] = max_iter
        self.Mwsolver.parameters["relative_tolerance"] = rel_tol
        self.Mwsolver.parameters["error_on_nonconvergence"] = True
        self.Mwsolver.parameters["nonzero_initial_guess"] = False

    def _fTV(self, m:dl.Function)->dl.Function:
        """Computes the TV functional.

        Args:
            m (dl.Function): Function to compute the TV norm of.

        Returns:
            dl.Function: TV norm of m.
        """
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
    
    
    def cost(self, m):
        # (smoothed) TV functional
        return self.alpha * dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)*dl.dx
    
    
    def grad(self, m, out):
        TVm = self._fTV(m)
        grad_tv = self.alpha * dl.Constant(1.)/TVm*dl.inner(dl.grad(m), dl.grad(self.m_tilde))*dl.dx
        
        # assemble the UFL form to a vector, add to out
        v = dl.assemble(grad_tv)
        out.axpy(1., v)
    
    
    def hess(self, m, w):
        TVm = self._fTV(m)
        
        # symmetrized version of (5.2) from [1]
        A = dl.Constant(1.)/TVm * ( dl.Identity(2) 
                                    - dl.Constant(0.5)*dl.outer(w, dl.grad(m)/TVm)
                                    - dl.Constant(0.5)*dl.outer(dl.grad(m)/TVm, w) )
        
        return self.alpha * dl.inner(A*dl.grad(self.m_tilde), dl.grad(self.m_hat))*dl.dx
    
    
    def compute_w_hat(self, m, w, m_hat):
        TVm = self._fTV(m)
        
        # symmetrized version of operator in (3.6) from [1]
        A = dl.Constant(1.)/TVm * ( dl.Identity(2) 
                                   - dl.Constant(0.5)*dl.outer(w, dl.grad(m)/TVm)
                                   - dl.Constant(0.5)*dl.outer(dl.grad(m)/TVm, w) )
        
        # expression for incremental slack variable (3.6) from [1]
        dw = A*dl.grad(m_hat) - w + dl.grad(m)/TVm
        dw = dl.assemble( dl.inner(self.w_tilde, dw)*dl.dx )
        
        # project into appropriate space
        out = dl.Vector(self.Mw.mpi_comm())
        self.Mw.init_vector(out, dw)
        
        return out
    
    
    def wnorm(self, w):
        # todo: assembly, projection
        return dl.inner(w, w)
    
    
class TVGaussianPrior:
    # primal implementation
    raise NotImplementedError("Fused TV+Gaussian Prior not yet implemented.")
