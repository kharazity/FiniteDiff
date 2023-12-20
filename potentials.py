import numpy as np

def PotentialOp(N, func):
    """
    N: The number of nodes 
    func: a function of a single variable. 
    func needs to be a 1d object, if other parameters need to be specified to
    define the function, one needs to use partial functions to set those arguments
    and pass the new function with the desired arguments for those parameters
    for example: LJd = partial(LJ, delta = d), you would pass LJd to 
    this funciton
    """
    return np.diag([func(i/(N-1)) for i in range(N)])

def LJ(r, delta = 0,sigma = 1,eps=1):
    return 4*eps*((sigma/(r+delta))**(12) - (sigma/(r + delta))**(6));

def Morse(r, r_e = 1, D=1):
    return D*(1-exp(-(abs(r)-r_e))**2)

def Coulomb(r,delta=0):
    return (r+delta)**(-2)

def Harmonic(r):
    return (r)**2

def PolyN(r,n):
    return (r)**n

def DLeft(N, Func, bcs = None, p = 1):
    """
    This function produces a diagonal matrix that approximates the function $\nabla V$ evaluated on a finite lattice of points and the derivative
    approximated with a finite difference scheme of order $p$. The diagonal matrix is just the matrix formed from placing the values of the vector V_i
    into a matrix V_{ii}
    N is the number of grid points
    Func, is the potential function we wish to evaluate
    bcs = boundary conditions, can be periodic, dirichlet, and Non-simple dirichlet,
    p = order of finite difference stencil used
    """
    return None
