import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial


from FDCoeffs import *
from potentials import *
from Unitaries import *

 
def ForwardDiffPer(N, p = 1):
    mat = np.zeros((N,N))
    coeffs = FDForwardCoeffs(p)
    shifts = np.array([shiftOps(N,i) for i in range(0,2*p)])
    i = 0
    for c in coeffs:
        mat += c*shifts[i]
        i += 1
    return mat

def BackwardDiffPer(N, p = 1):
    mat = np.zeros((N,N))
    coeffs = FDBackwardCoeffs(p)
    shifts = np.array([shiftOps(N,i) for i in range(-2*p+1,1)])
    i = 0
    for c in coeffs:
        mat += c*shifts[i]
        i += 1
    return mat

def ForwardDiffDir(N, p = 1):
    mat = np.zeros((N,N))
    coeffs = FDForwardCoeffs(p)
    shifts = np.array([shiftOpsNP(N,i) for i in range(0,2*p)])
    i = 0
    for c in coeffs:
        mat += c*shifts[i]
        i += 1
    return mat

def BackwardDiffDir(N, p = 1):
    mat = np.zeros((N,N))
    coeffs = FDForwardCoeffs(p)
    shifts = np.array([shiftOpsNP(N,i) for i in range(-2*p+1,1)])
    i = 0
    for c in coeffs:
        mat += c*shifts[i]
        i += 1
    return mat

def ForwardDiffNS(N, idxs, p = 1):

    return 

def BackwardDiffNS(N,idxs, p = 1):

    return

def LapPer(N,p=1):
    """
        Prepares the periodic Laplacian operator on N gridpoints with a 2p+1 central finite difference stencil
    """
    coeffs = FDCentralCoeffs(p)
    mat = sum(coeffs[i]*shiftOps(N, i) for i in range(1,p+1))
    mat+= coeffs[0]*np.identity(N)
    mat += sum(coeffs[i]*shiftOps(N, -i) for i in range(1, p+1))
    return mat

def LapDir(N, p=1):
    """
        Prepares the 1D Laplacian operator with Dirichlet boundary with a 2p+1 finite difference stencil
    """
    coeffs = FDCentralCoeffs(p)
    mat = sum(coeffs[i]*shiftOpsNP(N, i) for i in range(1,p+1))
    mat+= coeffs[0]*np.identity(N)
    mat += sum(coeffs[i]*shiftOpsNP(N, -i) for i in range(1, p+1))

    return mat


def LapDirNS(N, idxs, p=1, per = False):
    """
    Prepares the 1D Laplacian operator for Dirichlet boundaries on non-simply connected
    domains. Such as a 1D line with a hole in the middle. if per, then we assume that the 
    "edges" of the domain are connected. I'm actually not sure this makes a difference.
    """
    idxs1 = idxs.copy()
    idxs1.append(0)
    idxs1.append(idxs[-1]+1)
    
    idxs2 = idxs.copy()
    idxs2.append(N-1)
    idxs2.append(idxs[0]-1)
    mat = -2*np.identity(N) + .5*(shiftOps(N,1) + shiftOps(N,1)@refOp(N, idxs2) + shiftOps(N,-1) + shiftOps(N,-1)@refOp(N, idxs1) )
    #return .5*(mat + refOp(N, idxs[1: len(idxs)-1])@mat)
    return mat


def Lap1d(N, bcs, p =1, idxs = None):
    """
    idxs are the nodes where boundary conditions are enforced
    For example, for end point boundary conditions you'd input [0,N-1], with the left most-boundary point entered first in the list
    idxs[0] < idxs[1] for this to work.
    Now, if there is a set of boundary conditions that are enforced on the interior of the region, you'd input that list of nodes that are not a part of the computational domain
    and the end points of that list are considered as the boundary points, and set the points on the interior to cancel out the corresponding elements of the operator on those points  
    """
    if idxs is None:
        idxs = [0,N-1]
    if bcs == 'per':
        return LapPer(N,p)
    if bcs == 'dir':
        return LapDir(N, p)
    if bcs == 'dirNS':
        return LapDirNS(N, idxs)

def ConvectiveTerm1d(N, bcs, func, p = 1, idxs = None):
    if idxs is None:
        idxs = [0, N-1]
    if bcs == 'per':
        return ForwardDiffPer(N,p)@PotentialOp(N,func=func)@BackwardDiffPer(N,p)
    if bcs == 'dir':
        return ForwardDiffDir(N,p)@PotentialOp(N,func = func)@BackwardDiffDir(N,p)
    if bcs == 'dirNS':
        return 

def kronSum(*mats):
    """
    *mats allows for a variable number of matrices as argument
    This prepares the matrix that is the kronecker sum of all
    the matrices passed to the function
    """
    dims = [mat.shape[0] for mat in mats]
    leftDims = lambda i : int(np.prod(dims[0:i]))
    rightDims = lambda i : int(np.prod(dims[i+1:]))
    total_dim = np.prod(dims)
    sum = np.zeros((total_dim,total_dim))
    i = 0
    for mat in mats:
        if i == 0:
            sum += np.kron(mat, np.identity(rightDims(0)))
        elif i == len(dims):
            sum += np.kron(np.identity(leftDims(-1)), mat)
        else:
            sum += np.kron(np.kron(np.identity(leftDims(i)),mat),np.identity(rightDims(i)))
        i+=1
    return sum


""" 
TODO: Get order-p stencil working for the non-simple Dirichlet boundary conditions. 
Requires figuring out which indices are needed to be flipped to cancel the 
off diagonal elements in the middle.

Get two hole BCs implemented. Is there a simple way to automate the construction of this sum:
Dx = (np.kron(Lap1d(Nx, 'per'), sigMat(Ny, B1)) + np.kron(Lap1d(Nx, 'dirNS', A2), sigMat(Ny, B2)) + np.kron(Lap1d(Nx, 'per'), sigMat(Ny, B3)))
Dy = (np.kron(sigMat(Nx, A1), Lap1d(Ny, 'per')) + np.kron(sigMat(Nx, A2), Lap1d(Ny, 'dirNS', B2)) + np.kron(sigMat(Nx, A3), Lap1d(Ny, 'per')))
L2 = .5*(Dx + Dy)
What information is needed by the user to produce this automatically?

Get GradVGrad term of order p implemented with appropriate boundary conditions. For example, we would probably need to perform an LCU of V operators as:
V' = V + V@refOp(N, bndryIdxs), so that the rows that have only a diagonal entry have still the value -2H(n,p) from the Laplacian. Additionally, the 
gradient operators themselves will need to be constructed to have the entries cancel on the super and sub diagonals on those same rows.

"""


N = 16
D1 = ForwardDiffDir(N,2)
D2 = BackwardDiffDir(N,2)
plt.imshow(D1@D2)
plt.show()

L = LapDir(N, p=5)
plt.imshow(L)
plt.show()

plt.imshow(L - D1@D2)
plt.show()
d = .5*1/N
coul = partial(Coulomb, delta = d)
Vlj = partial(LJ, delta = d)
