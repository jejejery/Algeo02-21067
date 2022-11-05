import os
import numpy as np
import cv2
from time import *
from sklearn import preprocessing

dir = r'C:\\Users\\ASUS\\Documents\\Programming\\Python\\Algeo02-21067\\dataset\\train\\5\\BnW'


def labelimg(dir):
    label = []
    res = os.listdir(dir)
    for k in res:
        label.append(k.replace('.jpg',''))
    return label

def norm_img(image):
    newimg = np.delete(image,[1,2],2)
    return newimg.reshape(256*256)

def set_training(dir):
    training = np.ndarray(shape=(25,256*256))
    res = os.listdir(dir)
    ctr = 0
    for k in res:
        img = norm_img(cv2.imread(dir+'\\'+k))
        training[ctr] = img
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
    A = np.ndarray(shape=(25, 256*256))
    avg = avg_image(dir)
    res = os.listdir(dir)
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

def eigen(C):
    return np.linalg.eig(C)

def eigenSort(eigenval,eigenvec):
    eig_pairs = [(eigenval[index], eigenvec[:,index]) for index in range(len(eigenval))]
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenval))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenval))]
    return eigvalues_sort,eigvectors_sort

def eigenValMinorCoff(M,iterasi):
    for i in range(iterasi):
        Q, R = np.linalg.qr(M)
        M = R @ Q
    return np.diag(M)






    
    


#print(w.shape)
# u = np.array([])
# for i in range(10):
#     k = A.T@v[i]
#     u = np.concatenate((u,k),axis=0)

# u = u.reshape((10,int(len(u)/10)))
# print(u)








    

        




    

