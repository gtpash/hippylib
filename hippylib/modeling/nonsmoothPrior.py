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
    # primal-dual implementation for (vector) total variation prior
    def __init__(self, Vhm:dl.FunctionSpace, Vhw:dl.FunctionSpace, alpha:float, beta:float, rel_tol:float=1e-12, max_iter:int=100):
        self.alpha = dl.Constant(alpha)
        self.beta = dl.Constant(beta)
        self.Vhm = Vhm
        self.Vhw = Vhw

        # assemble mass matrix for parameter
        mtrial = dl.TrialFunction(Vhm)
        mtest  = dl.TestFunction(Vhm)
        
        varfM = ufl.inner(mtrial, mtest)*ufl.dx
        self.M = dl.assemble(varfM)
        self.Msolver = PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False
        
        # assemble mass matrix for slack variable
        wtrial = dl.TrialFunction(Vhm)
        wtest  = dl.TestFunction(Vhm)
        
        varfMw = ufl.inner(wtrial, wtest)*ufl.dx
        self.Mw = dl.assemble(varfMw)
        self.Mwsolver = PETScKrylovSolver(self.Vhw.mesh().mpi_comm(), "cg", "jacobi")
        self.Mwsolver.set_operator(self.M)
        self.Mwsolver.parameters["maximum_iterations"] = max_iter
        self.Mwsolver.parameters["relative_tolerance"] = rel_tol
        self.Mwsolver.parameters["error_on_nonconvergence"] = True
        self.Mwsolver.parameters["nonzero_initial_guess"] = False

    def _fTV(self, m:dl.Function)->dl.Function:
        """Helper function to compute the TV norm of a function m.

        Args:
            m (dl.Function): Function to compute the TV norm of.

        Returns:
            dl.Function: TV norm of m.
        """
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
    
    
    def cost(self, m):
        # (smoothed) TV functional
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)*dl.dx
    
    
    def grad(self, m, out):
        raise NotImplementedError("Gradient not implemented.")
    
    
class TVGaussianPrior:
    # primal implementation
    raise NotImplementedError("Fused TV+Gaussian Prior not yet implemented.")
