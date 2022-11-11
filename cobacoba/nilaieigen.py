import numpy as np
import time as t


def proj(u, v):
    v_norm_squared = sum(v**2)   
    
    proj_of_u_on_v = (np.dot(u, v)/v_norm_squared)*v
    return proj_of_u_on_v

def gram_schmidt(Q):
    lenRow = Q.shape[0] 
    lenCol = Q.shape[1] 
    newQ = Q.T
    X = np.ndarray(shape=(lenCol, lenRow), dtype=float)
    
    
    for i in range(lenCol):
        Y =  newQ[i]
        
        if((i != 0)):
            for j in range(i):
                Y -= proj(newQ[i], X[j])
        X[i] = Y
        

    for i in range(lenCol):
        X[i] = X[i]/np.linalg.norm(X[i])
    return X

def QR_decomposition(A):
    Z = np.array(A)
    Q = gram_schmidt(Z)
    R = np.dot(Q, A)
    return normNearZero(Q.T,1e-15), normNearZero(R,1e-15)

def normNearZero(A, epsilon):
    A.real[abs(A.real) < epsilon]  = 0
    return A


A = np.array([[1.0,2.0,3.0],[-1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])
  

Q,R = QR_decomposition(A)
Qi,eR = np.linalg.qr(A)
print("+++++++++++++++++=")
print("NILAI Q: ")
print(Q)
print("+++++++++++++++++=")
print("NILAI R: ")
print(R)
print("+++++++++++++++++=")

print("+++++++++++++++++=")
print("NILAI Qi: ")
print(Qi)
print("+++++++++++++++++=")
print("NILAI eR: ")
print(eR)



 

