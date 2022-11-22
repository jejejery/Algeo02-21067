import numpy as np

"""
******************************************************************************
-----------------------------KAKAS DEKOMPOSISI QR-----------------------------
******************************************************************************
"""
def proj(u, v):
    v_norm_squared = sum(v**2)   
    
    proj_of_u_on_v = (np.dot(u, v)/v_norm_squared)*v
    return proj_of_u_on_v

def norm_vector(v):
    return np.sqrt(np.sum(np.power(v,2)))
    #JIKA PAKAI FOR LOOP, TIDAK EFISIEN WAKTU

def cosine_sim(a,b):
    return abs(np.dot(a,b)/(norm_vector(a)*norm_vector(b)))

#Kalkulasi dekomposisi QR berdasarkan algoritma Schwarz-Rutishauser
def QR_decomposition(M, type='float64'):
    
    M = np.array(M, dtype=type)
    (m,n) = np.shape(M)

    Q = np.array(M, dtype=type)      
    R = np.zeros((n, n), dtype=type) #langkah 1

    for k in range(n):
        for i in range(k):
            R[i,k] = np.transpose(Q[:,i]).dot(Q[:,k]) #langkah 2, perbarui tiap elemen R
            Q[:,k] = Q[:,k] - R[i,k] * Q[:,i] #langkah 3, ortogonalisasi

        R[k,k] = norm_vector(Q[:,k]) #langkah 4a, update R[k][k]
        Q[:,k] = Q[:,k] / R[k,k] #langkah 4b, ortonormalisasi
    
    return -Q, -R   


