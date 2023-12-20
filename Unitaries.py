import numpy as np
import scipy as sp
from scipy.fft import dct, idct, dst, idst, fft, ifft


def DFT(N):
    return fft(sp.eye(N), axis = 0)

def iDFT(N):
    return ifft(sp.eye(N), axis = 0)

def DCT(N):
    return dct(np.eye(N), axis = 0)

def iDCT(N):
    return idct(np.eye(N), axis = 0)

def DST(N):
    return dst(np.eye(N), axis = 0)

def iDST(N):
    return idst(np.eye(N), axis = 0)


def shiftOps(N, pow):
    """
    with pow = j,
    Prepares the NxN unitary matrix that performs the action
    S(N,j)|i> = |(i+j)%N>, j in {-N+1, ..., 0, ..., N+1}. S(N,0) is the 
    NxN identity
    """
    arr = np.zeros((N,N))
    for i in range(N):
        arr[(i+pow)%N, i] = 1
    return arr

def shiftOpsBCs(N, pow, idxs = None):
    if idxs == None:
        idxs = [0, N-1]
    op = .5*shiftOps(N, pow)
    op += .5*shiftOps(N, pow)@refOp(N, [(x-pow)%N for x in idxs])
    return op

def refOp(N,idxs):
    """
    This prepares the diagonal reflection operator I - 2*sum(i in idxs)|i><i|, which is 1 on the
    locations in [0,...,N-1]\{idxs} and -1 on the locations in idxs
    """
    arr = np.identity(N)
    for i in idxs:
        arr[i,i] += -2
    return arr

def sigMat(N, idxs):
    """
        This prepares the matrix with ones on the diagonal elements in idxs
    """
    return .5*(np.identity(N) - refOp(N,idxs))


Zp = np.array([[1,0],[0,-1]])
def Znj(n,j):
    """
        Constructs the pauli Z operator on n qubits acting on the jth site
    """
    Inj = np.eye(2**(j-1))
    In = np.eye(2**(n-j))
    return np.kron(np.kron(In, Zp),Inj)
