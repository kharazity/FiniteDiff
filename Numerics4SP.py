#%%
from FDmats import *
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt


def uniform(N):
    return 1/np.sqrt(N)*np.ones(N)

for n in range(2,5):
    N = 2**n
    L = Lap1d(N, bcs="dir")
    #L *= 1/norm(L, ord = 1)
    rhs = np.zeros(N)
    rhs[N-1] = -1.
    b = L@rhs;
    print(np.linalg.norm(b)**2)

#%%

# arr = np.zeros(6)
i = 0
Narr = np.arange(120,121)
arr = np.zeros(len(Narr))
for N in Narr:
    L = Lap1d(N, bcs= 'dir')
    L2 = kronSum(L,L)
    L2 *= 1/norm(L2, ord = 1)
    rhs = np.zeros(N)
    rhs[N-1] = -2.
    b = np.kron(rhs,uniform(N))
    SP = norm(L2@b,ord=2)
    arr[i] = SP**2
    i+=1
plt.plot(Narr,arr)
print(arr)



#%%
print("\n")
for n in range(7,8):
    N = 2**n
    L = Lap1d(N, bcs="dir")
    Ly = Lap1d(N, bcs='per')
    L2 = kronSum(L,Ly)
    L2 *= 1/norm(L2, ord = 1)
    rhs = np.zeros(N)
    rhs[N-1] = 1.
    rhs2 = np.kron(rhs,uniform(N))
    print(norm(L2@rhs2))

#%%


print("\n")
for n in range(5,6):
    N = 2**n
    L = Lap1d(N, bcs="dir")
    L3 = kronSum(L,L,L)
    L3 *= 1/norm(L3,ord=1)
    rhs = np.zeros(N)
    rhs[N-1] = -1.
    rhs2 = np.kron(np.kron(rhs,uniform(N)),uniform(N))
    print(norm(L3@rhs2))

# %%
