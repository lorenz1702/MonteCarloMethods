import numpy as np

def init_config(N):
    L = np.full((N,N),1)
    return L

def get_E(L):
    E=0
    for i in range(N):
        for j in range (N):
            E-=L[i][j]*(L[(i+1)%N][j]+L[(i-1)%N][j]+L[i][(j+1)%N]+L[i][j-1])
    return E

def prob_dis(Delta_E):

    return np.exp(-beta*Delta_E)

def Delta_E(L,x):
    Delta_E=2*L[i][j]*(L[(i+1)%N][j]+L[(i-1)%N][j]+L[i][(j+1)%N]+L[i][j-1])
    return Delta_E

def thermalization(L):

    for i in range(N):
        for j in range (N):
            Delta_E=Delta_E(L,(i,j))
            if (Delta_E<=0):
                L[i][j]=(-L[i][j])
            else:
                p=np.random.uniform(0,1)
                if (p<=prob_dis(Delta)):
                    L[i][j]=(-L[i][j])
                
    