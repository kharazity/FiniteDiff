from PosOps import *
import numpy as np
from numpy.linalg import *
from Unitaries import *
from FDmats import *
from potentials import *
import matplotlib.pyplot as plt


n = 3
N = 2**n
XI = np.kron(oneBodyPosOp(N), np.eye(N))
IX = np.kron(np.eye(N), oneBodyPosOp(N))
D1m = np.kron(BackwardDiffPer(N), np.eye(N))
D2m = np.kron(np.eye(N),BackwardDiffPer(N))

Dx1 = D1m@XI
plt.imshow(Dx1)
plt.show()

Dx2 = D1m@IX
plt.imshow(Dx2)
plt.show()

