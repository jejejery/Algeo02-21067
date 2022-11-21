import os
import numpy as np
import cv2
from time import *
from PIL import Image


"""
******************************************************************************
-------------------------------------FUNGSI UTAMA-----------------------------
******************************************************************************
"""

def training_parameters(training):
    #Menghitung kovarians dari data training
    avg = avg_image(training)
    A = training - avg
    kov = covariance(A)

    #menghitung eigenface dan weight
    eigval, eigenvec = theEigen(kov,100)
    eigval, eigenvec = eigenSort(eigval,eigenvec)
    eigenvec = np.array(eigenvec)
    x = int(training.shape[0]*0.75)
    eigfaces = np.array(get_eigenfaces(eigenvec,x,training)) #100%
    weight_training = get_weight(eigfaces,A)

    return eigfaces, weight_training, avg

def metrics_calculation(test,eigenfaces,weight_training,avg_training,label_training):
    weight_test = np.dot(eigenfaces,test-avg_training)
    ctr = 0
    
    for i in weight_training:
        if ctr == 0:
            euclidian_distance = np.linalg.norm(weight_test-i)
            cos_sim = cosine_sim(weight_test,i)
            mark = 0
        else:
            if np.linalg.norm(weight_test-i) < euclidian_distance:
                euclidian_distance = np.linalg.norm(weight_test-i)
                mark = ctr
                cos_sim = cosine_sim(weight_test,i)
        ctr += 1

    
    return euclidian_distance, cos_sim, label_training[mark]

"""
******************************************************************************
-----------------------------KAKAS PENGOLAHAN CITRA---------------------------
******************************************************************************
"""

def labelimg(dir):
    label = []
    res = os.listdir(dir)
    for k in res:
        label.append(k.replace('.jpg',''))
    return label

def norm_img(image):
    dim = (256,256)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    return resized.reshape(256*256)

def get_img(img):
    dim = (256, 256)
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    return resized.reshape(256*256)

def get_img_PIL(img):
    theImg = norm_img(np.array(Image.open(img)))
    return theImg
    

def set_training(dir):
    res = os.listdir(dir)
    training = np.ndarray(shape=(len(res),256*256))
    ctr = 0
    for k in res:
        image = cv2.imread(dir+'\\'+k)
        rshp = norm_img(image)
        training[ctr] = rshp
        ctr += 1
    return training

def avg_image(training_set):
    s = np.zeros((256*256))
    for k in range(training_set.shape[0]):
        s = s + training_set[k]
    
    return s/(training_set.shape[0])

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

def get_eigenfaces(eigenvectors,n,training_tensor):
    reduced_data = np.array(eigenvectors[:n]).transpose()
    eigenface = np.dot(training_tensor.T,reduced_data)
    return eigenface.T


def get_weight(eigenfaces,norm_training_set):
    return np.array([np.dot(eigenfaces,k) for k in norm_training_set])

"""
******************************************************************************
-----------------------------KAKAS DEKOMPOSISI QR-----------------------------
******************************************************************************
"""
def proj(u, v):
    v_norm_squared = sum(v**2)   
    
    proj_of_u_on_v = (np.dot(u, v)/v_norm_squared)*v
    return proj_of_u_on_v

def cosine_sim(a,b):
    return abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def QR_decomposition(M, type='float64'):
    
    M = np.array(M, dtype=type)
    (m,n) = np.shape(M)

    Q = np.array(M, dtype=type)      
    R = np.zeros((n, n), dtype=type)

    for k in range(n):
        for i in range(k):
            R[i,k] = np.transpose(Q[:,i]).dot(Q[:,k])
            Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]

        R[k,k] = np.linalg.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k]
    
    return -Q, -R   











    

        




    

