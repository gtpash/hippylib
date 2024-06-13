import dolfin as dl
import ufl
from petsc4py import PETSc

# todo: add preconditioner for jacobian
# todo: better prefix handling
# todo: potentially add an options manager to set PETSc options

class SNES_VariationalProblem():
    """Direct use of PETSc SNES interface to solve
    Nonlinear Variation Problem F(u; v) = 0.
    ref: https://fenicsproject.discourse.group/t/dusing-petsc4py-petsc-snes-directly/2368/18
    ref: https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/4
    """
    def __init__(self, F:ufl.Form, u:dl.Function, bcs:list):
        """Constructor.

        Args:
            F (ufl.Form): The nonlinear form (typically a PDE in residual form)
            u (dl.Function): The function to solve for.
            bcs (list): List of boundary conditions.
        """
        self.u = u                                  # initial iterate
        self.V = self.u.function_space()
        du = dl.TrialFunction(self.V)
        self.u_test = dl.TestFunction(self.V)
        self.L = F                                  # residual form of nonlinear problem
        self.J_form  = dl.derivative(F, u, du)      # Jacobian form
        self.bcs = bcs

    def F(self, snes, x, F):
        """Form the residual for this problem.

        Args:
            snes (PETSc.SNES): PETSc SNES object.
            x (PETSc.Vec): Current iterate.
            F (PETSc.Vec): Residual at current iterate.
        """
        x = dl.PETScVector(x)
        Fvec  = dl.PETScVector(F)
        
        x.vec().copy(self.u.vector().vec())     # copy PETSc iterate to dolfin
        self.u.vector().apply("")               # update ghost values
        dl.assemble(self.L, tensor=Fvec)        # assemble residual
        
        # apply boundary conditions
        for bc in self.bcs:
            bc.apply(Fvec, x)
            bc.apply(Fvec, self.u.vector())

    def J(self, snes, x, J, P):
        """Form the residual for this problem.

        Args:
            snes (PETSc.SNES): PETSc SNES object.
            x (PETSc.Vec): Current iterate.
            F (PETSc.Vec): Residual at current iterate.
            P (PETSc.Vec): Preconditioner.
        """
        J = dl.PETScMatrix(J)
        x.copy(self.u.vector().vec())       # copy PETSc iterate to dolfin
        self.u.vector().apply("")           # update ghost values
        dl.assemble(self.J_form, tensor=J)  # assemble Jacobian

        # apply boundary conditions.
        for bc in self.bcs:
            bc.apply(J)
            #  bc.apply(P)


class SNES_VariationalSolver():
    """Direct use of PETSc SNES interface to solve
    Nonlinear Variation Problem F(u; v) = 0.
    
    See the following forum posts for more information.
    ref: https://fenicsproject.discourse.group/t/dusing-petsc4py-petsc-snes-directly/2368/18
    ref: https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/4
    """
    def __init__(self, problem, comm):
        """Constructor.

        Args:
            problem (SNES_VariationalProblem): The nonlinear variational problem.
            comm: An MPI communicator.
        """
        self.problem=problem
        self.comm=comm

        # create SNES solver
        self.snes = PETSc.SNES().create(self.comm)
        self.snes.setFromOptions()

        # set function, jacobian
        b = dl.PETScVector()
        J_mat = dl.PETScMatrix()
        # P_mat = dl.PETScMatrix()
        self.snes.setFunction(self.problem.F, b.vec())
        self.snes.setJacobian(problem.J, J_mat.mat())

    def set_pc_IS(self):
        r""" set preconditioner index set - ignore if using mumps"""
        pc=self.snes.ksp.pc
        fields=[]
        # index set for pressure components, e.g. V.sub(0)
        i=0
        subdofs=self.problem.V.sub(i).dofmap().dofs()
        IS=PETSc.IS().createGeneral(subdofs)
        fields.append((str(i), IS))
        pc.setFieldSplitIS(*fields)

    def solve(self):
        self.snes.solve(None, self.problem.u.vector().vec())

    def getConvergedReason(self):
        return self.snes.getConvergedReason()

    def getIterationNumber(self):
        return self.snes.getIterationNumber()
