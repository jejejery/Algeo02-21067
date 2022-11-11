import os
import numpy as np
import cv2
from time import *
from sklearn import preprocessing


dir = r'C:\\Users\\ASUS\\Documents\\Programming\\Python\\Algeo02-21067\\dataset\\train'


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

def get_test(dir):
    res = os.listdir(dir)
    dim = (256, 256)
    test_set = np.ndarray(shape=(len(res),256*256))
    ctr = 0
    for k in res:
        image = cv2.imread(dir+'\\'+k)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
        test_set[ctr] = resized.reshape(256*256)
        ctr += 1
    return test_set

def get_img(img):
    dim = (256, 256)
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    return resized.reshape(256*256)
    

def set_training(dir):
    dim = (256,256)
    res = os.listdir(dir)
    training = np.ndarray(shape=(len(res),256*256))
    ctr = 0
    for k in res:
        image = cv2.imread(dir+'\\'+k)
        rshp = norm_img(image)
        training[ctr] = rshp
        ctr += 1
    return training



def avg_image(dir):
    res = os.listdir(dir)
    s = np.zeros((256*256))
    for k in res:
        img = cv2.imread(dir+'\\'+k)
        s = s + norm_img(img)
    
    return s/(len(res))
    

def A_Matrix(dir):
    avg = avg_image(dir)
    res = os.listdir(dir)
    A = np.ndarray(shape=(len(res), 256*256))
    ctr = 0
    for k in res:
        img = norm_img(cv2.imread(dir+'\\'+k)) - avg
        A[ctr] = img
        ctr += 1
    # A = A.reshape(256*256,len(res))
    # A = A.astype('float32')
    return  A 

def covariance(A):
    return A @ A.T

def eigenNP(C):
    return np.linalg.eig(C)

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
        Y = R @ Q #np.dot(R,Q)
        vec = vec @ Q
    

    return Y.diagonal(),vec 



def proj(u, v):
    v_norm_squared = sum(v**2)   
    
    proj_of_u_on_v = (np.dot(u, v)/v_norm_squared)*v
    return proj_of_u_on_v

def cosine_sim(a,b):
    return abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def get_eigenfaces(eigenvectors,n):
    reduced_data = np.array(eigenvectors[:n]).transpose()
    training_tensor = set_training(dir)
    eigenface = np.dot(training_tensor.T,reduced_data)
    return eigenface.T

def get_weight(eigenfaces,norm_training_set):
    return np.array([np.dot(eigenfaces,k) for k in norm_training_set])

def proj(u, v):
    v_norm_squared = sum(v**2)   
    
    proj_of_u_on_v = (np.dot(u, v)/v_norm_squared)*v
    return proj_of_u_on_v




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


#print(w.shape)
# u = np.array([])
# for i in range(10):
#     k = A.T@v[i]
#     u = np.concatenate((u,k),axis=0)

# u = u.reshape((10,int(len(u)/10)))
# print(u)








    

        




    

