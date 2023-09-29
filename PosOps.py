import numpy as np
from functools import partial


from FDCoeffs import *
from Unitaries import *

def oneBodyPosOp(N):
    """
        Follows the construction given in Lemma 24 of Quantum Simulatoin of the First-Quantized Pauli-Fierz Hamiltonian.
        The subnormalization factor is alpha = N-1
    """
    n = int(np.log2(N))
    X = (2**n - 1)/2*np.eye(N) - 1/2*sum([2**j*Znj(n,j+1) for j in range(0,n)])
    X *= 1/(N-1)
    return X
