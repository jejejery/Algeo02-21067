import os
import numpy as np
import cv2
from time import *
from sklearn import preprocessing
from calculate import *

dir = r'C:\\Users\\ASUS\\Documents\\Programming\\Python\\Algeo02-21067\\dataset\\train\\5\\BnW'




A = A_Matrix(dir)
kov = covariance(A)
#Bagian yang harus kita kulik sendiri, ga boleh pakai library linear algebra
eigenvalues, eigenvectors, = eigen(kov)


# Sort the eigen pairs in descending order:

eigenvalues, eigenvectors = eigenSort(eigenvalues,eigenvectors)

# print('Eigenvectors of Cov(X): \n%s' %eigvectors_sort)
# print('\nEigenvalues of Cov(X): \n%s' %eigvalues_sort)


#------------------------#
#Menampilkan EigenFace

reduced_data = np.array(eigenvectors[:24]).transpose()

training_tensor = set_training(dir)
proj_data = np.dot(training_tensor.transpose(),reduced_data)
proj_data = proj_data.transpose()

img = proj_data[2].reshape(256,256).astype('uint8')

grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.imshow('image',grayImage)
cv2.waitKey(0)

cv2.destroyAllWindows()


#def eigenvector():
#print("Dengan dekomposisi QR: ",eigenval(C,20))
#print("Dengan library eigen dari numpy: ",w)

# def eigenVectorMajor(v,A):
#     container = np.array([])
#     for i in range (v.shape[0]):
#         temp = A @ v[i] 
#         container = np.concatenate((container,temp),axis=0)
#     container = container.reshape((int(len(container)/v.shape[0]),v.shape[0]))
#     return container


# average = avg_image(dir).reshape(256,256).astype('uint8')
# img = norm_img(cv2.imread(dir+'\\Chris Hemsworth111_393.jpg')).reshape(256,256)
# img = np.subtract(img,average)
# print(average)

# img = norm_img(cv2.imread(dir+'\\Chris Hemsworth111_393.jpg')).reshape(256,256)
# print(img.dtype)

# grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# cv2.imshow('image',grayImage)
# cv2.waitKey(0)

# cv2.destroyAllWindows()
# v_major = eigenVectorMajor(v,A)
# v_major = v_major.T
# print(v_major.shape)


# eigenfaces1 = preprocessing.minmax_scale(v_major[0], feature_range=(0,255))
# eigenfaces1 = v_major[3]
# eigenfaces1 = eigenfaces1.reshape(256,256).astype('uint8')

# print(eigenfaces1)


# grayImage = cv2.cvtColor(eigenfaces1, cv2.COLOR_GRAY2BGR)

# cv2.imshow('image',grayImage)
# cv2.waitKey(0)

# cv2.destroyAllWindows()




    
    


#print(w.shape)
# u = np.array([])
# for i in range(10):
#     k = A.T@v[i]
#     u = np.concatenate((u,k),axis=0)

# u = u.reshape((10,int(len(u)/10)))
# print(u)








    

        




    

