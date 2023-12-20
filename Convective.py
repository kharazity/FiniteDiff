from PosOps import *
import numpy as np
from numpy.linalg import *
from Unitaries import *
from FDmats import *
from potentials import *
import matplotlib.pyplot as plt

"""
What to do:
We need to introduce the $\delta$ parameter for the cutoff. 
We need a matrix corresponding to grad V @ grad
"""

"""
The potential function is a function of the absolute difference of position
V(x_i - x_j) = V(r_k)
where $k = iN + j
"""


def gradHarmonic(N, delta = 0.):
    X1 = oneBodyPosOp(N)
    X2 = oneBodyPosOp(N)
    return .5*kronSum(X1,-X2)


def centerDiff(N, bcs = None):
    if bcs == None:
        A = .5*(shiftOps(N, 1) - shiftOps(N,-1))
        return kronSum(A,A)
    else:
        A = .5*(shiftOpsBCs(N, 1, bcs) - shiftOpsBCs(N,-1, bcs))
        return kronSum(A,A)

def HarmonicConvectiveTerm(N, bcs = None):
    A = gradHarmonic(N)
    B = centerDiff(N, bcs)
    return A@B
N = 8
bcs = [0, N-1]
idxs = [bcs[0]*N+bcs[0], bcs[0]*N + bcs[1], bcs[1]*N + bcs[0], bcs[1]*N + bcs[1]]

V = gradHarmonic(N)
D = centerDiff(N, bcs = [0,N-1])
D1 = centerDiff(N)
B = V@D
"""
Sanity check, you can see that 
B[idxs, :] = 0's, which implies that the rows corresponding
to boundary nodes are all zero, which is the desired property
"""