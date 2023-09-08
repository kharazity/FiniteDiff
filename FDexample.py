from FDmats import *
import numpy as np
"""
#Example, 10 grid points in X and Y directions
# The box of points {(4,3), (6,3), (4,6), (6,6)} 
# is not part of the computational domain and correspond 
# to a hole removed from the region [1,...,Nx]x[1,...,Ny]

Ny - *---------------* (Nx,Ny)
|    |     ____      | 
*    |    |    |     |
|    |    |____|     |
*    |               |
|    |               |
*    *---------------*  Nx
     |---|-----|-----|
     A1   A2    A3
"""
Nx = 16
Nl = 6
Nr = 9
A1 = [x for x in range(Nl)]
A2 = [x for x in range(Nl,Nr)]
A3 = [x for x in range(Nr,Nx)]

Ny = Nx
B1 = A1
B2 = A2
B3 = A3

#The operator is formed by a sequence of 1d Laplacians

Dx = (np.kron(Lap1d(Nx, 'per'), sigMat(Ny, B1)) + np.kron(Lap1d(Nx, 'dirNS', A2), sigMat(Ny, B2)) + np.kron(Lap1d(Nx, 'per'), sigMat(Ny, B3)))
Dy = (np.kron(sigMat(Nx, A1), Lap1d(Ny, 'dir')) + np.kron(sigMat(Nx, A2), Lap1d(Ny, 'dirNS', B2)) + np.kron(sigMat(Nx, A3), Lap1d(Ny, 'per')))
L2 = .5*(Dx + Dy)

"""
#The boundary conditions are formed by rhs = 0 if the index is on the interior boundary nodes

#BC's =1 on [A2]x[B2]
x_vec = np.zeros(Nx)
x_vec[0] = 2
x_vec[-1] = 2

y_vec = np.zeros(Nx)
y_vec[0] = 2
y_vec[-1] = 2

"""

x_vec = np.zeros(Nx)
x_vec[A2[:]] = 2

y_vec = np.zeros(Ny)
y_vec[B2[:]] = 2

rhs_vec = -1.*np.kron(x_vec,y_vec)

q = np.linalg.solve(L2, rhs_vec)
plt.plot(rhs_vec)
plt.plot(q)
plt.show()
q2d = np.reshape(q, (Nx,Ny))

X = np.arange(0,1, 1/Nx)
Y = np.arange(0,1, 1/Ny)
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
axes = fig.gca(projection ='3d')
axes.plot_surface(X, Y, q2d)
  
plt.show()


"""

d = .5*1/N
coul = partial(Coulomb, delta = d)
Vlj = partial(LJ, delta = d)

L1 = (shiftOps(N,1) - shiftOps(N,-1))@PotentialOp(N, coul)

L2 = ForwardDiffPer(N)@PotentialOp(N, coul)@BackwardDiffPer(N)

"""