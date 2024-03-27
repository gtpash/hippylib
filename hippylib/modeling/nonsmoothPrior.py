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

class TVPrior:
    def __init__(self, Vh:dl.FunctionSpace, alpha:float, beta:float, LD:bool=False):
        self.alpha = dl.Constant(alpha)
        self.beta = dl.Constant(beta)
        self.Vh = Vh
        self.LD = LD

    def _fTV(self, m:dl.Function)->dl.Function:
        """Helper function to compute the TV norm of a function m.

        Args:
            m (dl.Function): Function to compute the TV norm of.

        Returns:
            dl.Function: TV norm of m.
        """
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
    
    
    def cost(self, m):
        raise NotImplementedError("Cost not implemented.")
    
    
    def grad(self, m, out):
        raise NotImplementedError("Gradient not implemented.")
    
    
class TVGaussianPrior:
    raise NotImplementedError("Fused TV+Gaussian Prior not yet implemented.")
