import numpy as np
from .QR_Decomposition import *

"""
******************************************************************************
-----------------------------KAKAS KALKULASI EIGEN---------------------------
******************************************************************************
"""

def covariance(A):
    return A @ A.T


def eigenSort(eigenval,eigenvec):
    eig_pairs = [(eigenval[index], eigenvec[:,index]) for index in range(len(eigenval))]
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenval))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenval))]
    return eigvalues_sort,eigvectors_sort

def theEigen(M,iterasi):
    vec = np.identity(M.shape[0])
    Y = np.array(M)
    for i in range(iterasi):
        Q, R = QR_decomposition(Y)
        Y = R @ Q 
        vec = vec @ Q

    return Y.diagonal(),vec 

def get_eigenfaces(eigenvectors,n,training_set):
    reduced_data = np.array(eigenvectors[:n]).T
    eigenface = np.dot(training_set.T,reduced_data)
    return eigenface.T


def get_weight(eigenfaces,norm_training_set):
    return np.array([np.dot(eigenfaces,k) for k in norm_training_set])