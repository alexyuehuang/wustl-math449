#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *
from numpy.linalg import norm

def simpleOptim(A, b):
    '''
    input two matrices, output min value and where min is achieved. 
    use jacobi method
    assume we are optimizing the function for problem 3
    '''
    xmin=jacobiIteration(A,b)[-1]
    xtrans=np.transpose(xmin)
    btrans=np.transpose(b)
    first=(xtrans@A@xmin)
    fmin=first*0.5-btrans@xmin
    return fmin, xmin


def getAuxMatrix(A):
    '''
    return D, L, U matrices for Jacobi or Gauss-Seidel
    D: array
    L, U: matrices
    '''
    # m = A.shape[0]
    D = csr_matrix.diagonal(A)
    L = -tril(A, k=-1)
    U = -triu(A, k=1)
    return D, L, U
def jacobiIteration(A, b, x0=None, tol=1e-13, numIter=100):
    '''
    Jacobi iteraiton:
    A: nxn matrix
    b: (n,) vector
    x0: initial guess
    numIter: total number of iteration
    tol: algorithm stops if ||x^{k+1} - x^{k}|| < tol
    return: x
    x: solution array such that x[i] = i-th iterate
    '''
    n = A.shape[0]
    x = np.zeros((numIter+1, n))
    if x0 is not None:
        x[0] = x0
    D, L, U = getAuxMatrix(A)
    for k in range(numIter):
        x[k+1] = ((L+U)@x[k])/D + b/D
        if norm(x[k+1] - x[k]) < tol:
            break
    
    return x[:k+1]
# %%
def getMatrix(n=10, isDiagDom=True):
    '''
    Return an nxn matrix which is the discretization matrix
    from finite difference for a diffusion problem.
    To visualize A: use plt.spy(A.toarray())
    '''
    # Representation of sparse matrix and right-hand side
    assert n >= 2
    n -= 1
    diagonal  = np.zeros(n+1)
    lower     = np.zeros(n)
    upper     = np.zeros(n)

    # Precompute sparse matrix
    if isDiagDom:
        diagonal[:] = 2 + 1/n**2
    else:
        diagonal[:] = 2
    lower[:] = -1  #1
    upper[:] = -1  #1
    # Insert boundary conditions
    # diagonal[0] = 1
    # upper[0] = 0
    # diagonal[n] = 1
    # lower[-1] = 0

    
    A = diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(n+1, n+1),
        format='csr')

    return A
A = getMatrix(n = 10, isDiagDom=False)
b=np.ones(10)
fmin,xmin=simpleOptim(A,b)
print(fmin)
print(xmin)
# %%
