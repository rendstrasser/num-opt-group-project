import numpy as np
def scaling_ruiz(A):

    threshold = 1 + 10**-5

    m, n = A.shape
    
    d1 = np.ones(shape=m)
    d2 = np.ones(shape=n)
    
    B = A
    
    r1 = np.inf
    r2 = np.inf

    while r1 > threshold or r2 > threshold:
        row_norm = np.sum(np.abs(B)**2,axis=1)**(1/2)
        col_norm = np.sum(np.abs(B)**2,axis=0)**(1/2)
        
        d1 = np.multiply(d1, row_norm**(-1/2))
        d2 = np.multiply(d2, (m/n)**1/4 * col_norm**(-1/2))
        B = np.diag(d1) @ A @ np.diag(d2)
        
        row_norm = np.sum(np.abs(B)**2,axis=1)**(1/2)
        col_norm = np.sum(np.abs(B)**2,axis=0)**(1/2)
        
        r1 = np.max(row_norm) / np.min(row_norm)
        r2 = np.max(col_norm) / np.min(col_norm)
    
    D = np.diag(d1)
    E = np.diag(d2)

    return D, E
