import numpy as np
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

def shiftOpsBCs(N, pow, idxs):
    op = .5*shiftOps(N, pow)
    op += .5*shiftOps(N, pow)@refOp(N, idxs)
    return op

def shiftOpsNP(N, pow):
    op = shiftOps(N, pow)
    if pow == 0:
        return op
    elif pow > 0:
        op += shiftOps(N, pow)@refOp(N, [N - pow-1] + [x for x in range(N-pow, N)])
    else:
        op += shiftOps(N, pow)@refOp(N, [-pow]+[x for x in range(0,-pow)])
    return .5*op

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
