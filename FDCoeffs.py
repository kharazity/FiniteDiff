import findiff
import numpy as np
def HarmonicN(n,m):
    return np.sum(np.array([1/(k**m) for k in range(1,n+1)]));

def factorial(n):
    return np.prod(np.array([x for x in range(1,n+1)]))

def coeff(n,j):
    return (2*(-1)**(j+1)*(factorial(n))**2)/(j**2*factorial(n-j)*factorial(n+j))

def FDCentralCoeffs(p):
    """
    Returns the list of 2p+1 central finite difference coefficients for an order-p accurate stencil for 2nd derivative
    """
    coeffs = []
    for i in range(p+1):
        if i == 0:
            coeffs.append(-2*HarmonicN(p, 2))
        else:
            coeffs.append(coeff(p,i))
    return coeffs

def subNormFactor(p):
    coeffs = FDCentralCoeffs(p)
    s = 0
    for x in coeffs:
        s+= abs(x)
    return s

def FDForwardCoeffs(p):
    coeffs = findiff.coefficients(deriv=1,offsets=[x for x in range(0,2*p)], symbolic=False)['coefficients']
    return coeffs

def FDBackwardCoeffs(p):
    coeffs = findiff.coefficients(deriv=1,offsets=[x for x in range(-2*p+1,1)], symbolic=False)['coefficients']
    return coeffs
