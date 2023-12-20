import numpy as np
from numpy import linalg as LA
from FDmats import *
from Convective import *
from Unitaries import *
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

def fullMat(N, idxs):
    A = Lap1d(N, bcs= 'dir')
    A = kronSum(A,A)
    B = HarmonicConvectiveTerm(N,bcs = idxs)
    C = A + B
    return C


condsn = []
max = 100
for N in range(2,max,10):
    idxs = [0, N-1]
    condsn.append(LA.cond(fullMat(N, idxs)))

plt.plot([n**2 for n in range(2,max,10)], condsn,label = "Cond Num")
plt.plot([n**2 for n in range(2,max,10)],[x**2 for x in range(2,max,10)], label = r'$n$')
plt.legend()
plt.title('Condition num for Harmonic Potential 2 particles')
plt.xlabel('dimension')
plt.ylabel(r'${\lambda_{max}}/{\lambda_{min} }$')
plt.show()