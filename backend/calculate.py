import os
import numpy as np
import cv2
from time import *

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


def avg_image(dir):
    res = os.listdir(dir)
    s = np.zeros((256*256))
    for k in res:
        img = cv2.imread(dir+'\\'+k)
        s = s + norm_img(img)
    
    return s/(len(res))
    



def Cov_img(dir):
    A = np.array([])
    avg = avg_image(dir)
    res = os.listdir(dir)
    for k in res:
        img = norm_img(cv2.imread(dir+'\\'+k)) - avg
        A = np.concatenate((A,img),axis=0)
    A = A.reshape(len(res),256*256)
    A = A.astype('float32')
    return A

A = Cov_img(dir)
C = A @ A.T

#Bagian yang harus kita kulik sendiri, ga boleh pakai library linear algebra
w, v = np.linalg.eig(C)
    
#print(w.shape)
u = np.array([])
for i in range(10):
    k = A.T@v[i]
    u = np.concatenate((u,k),axis=0)

u = u.reshape((10,int(len(u)/10)))
print(u)






    

        




    

