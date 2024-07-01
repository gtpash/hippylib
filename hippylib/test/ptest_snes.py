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

import sys
import unittest
from mpi4py import MPI
import dolfin as dl

sys.path.append('../../')
import hippylib as hp

class TestSNES(unittest.TestCase):

    def test_snes_variational_problem(self):
            """Test Newton solver for a simple nonlinear PDE
            
            FEniCS 2019.1.0 version of the DolfinX example:
            https://github.com/FEniCS/dolfinx/blob/b6864c032e5e282f9b73f80523f8c264d0c7b3e5/python/test/unit/nls/test_newton.py#L190
            """
            from petsc4py import PETSc

            mesh = dl.UnitSquareMesh(MPI.COMM_WORLD, 12, 15)
            V = dl.FunctionSpace(mesh, "Lagrange", 1)
            u = dl.Function(V)
            v = dl.TestFunction(V)
            F = dl.inner(5.0, v) * dl.dx - dl.sqrt(u * u) * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx - dl.inner(u, v) * dl.dx
            
            bc = dl.DirichletBC(V, dl.Constant(1.), "on_boundary")
            
            problem = hp.SNES_VariationalProblem(F, u, [bc])
            u.assign(dl.Constant(0.9))  # initial guess
            
            b_vec = dl.PETScVector()
            J_mat = dl.PETScMatrix()
            
            snes = PETSc.SNES().create()
            snes.setFunction(problem.evalFunction, b_vec.vec())
            snes.setJacobian(problem.evalJacobian, J_mat.mat())
            
            snes.setTolerances(rtol=1.0e-9, max_it=10)
            snes.getKSP().setType("preonly")
            snes.getKSP().setTolerances(rtol=1.0e-9)
            snes.getKSP().getPC().setType("lu")

            snes.solve(None, problem.u.vector().vec())
            assert snes.getConvergedReason() > 0
            assert snes.getIterationNumber() < 6
            
            # Modify boundary condition and solve again
            bc = dl.DirichletBC(V, dl.Constant(0.6), "on_boundary")
            problem = hp.SNES_VariationalProblem(F, u, [bc])
            
            snes.solve(None, problem.u.vector().vec())
            assert snes.getConvergedReason() > 0
            assert snes.getIterationNumber() < 6
            # print(snes.getIterationNumber())
            # print(snes.getFunctionNorm())

            snes.destroy()