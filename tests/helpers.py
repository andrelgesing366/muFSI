"""Small numerical test doubles shared by active and legacy solver tests."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse


class MatrixHandle:
    """Only the serial PETSc matrix interface used by the response solver."""

    def __init__(self, matrix):
        self.matrix = sparse.csr_matrix(matrix)
        self.destroyed = False

    def getValuesCSR(self):
        return self.matrix.indptr, self.matrix.indices, self.matrix.data

    def getSize(self):
        return self.matrix.shape

    def destroy(self):
        self.destroyed = True


class VectorHandle:
    """Only the serial PETSc vector interface used by the response solver."""

    def __init__(self, values):
        self.values, self.destroyed = np.asarray(values), False

    def getArray(self, readonly=True):
        return self.values

    def destroy(self):
        self.destroyed = True


class TinyStructure:
    """Three structural DOFs with one constraint and caller-owned handles."""

    def __init__(self):
        self.K = np.array([[9.0, 1.0, 0.5], [1.0, 4.0, 0.25], [0.5, 0.25, 6.0]])
        self.M = np.diag([0.05, 0.1, 0.2])
        self.force = np.array([7.0, 1.0 + 0.2j, 0.4])
        self.constrained_dofs = np.array([0])
        self.mesh = SimpleNamespace(comm=SimpleNamespace(size=1))
        self.handles = []

    def stiffness_matrix(self):
        handle = MatrixHandle(self.K)
        self.handles.append(handle)
        return handle

    def mass_matrix(self):
        handle = MatrixHandle(self.M)
        self.handles.append(handle)
        return handle

    def force_vector(self, load):
        handle = VectorHandle(self.force)
        self.handles.append(handle)
        return handle
