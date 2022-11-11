import numpy as np
from time import *
from sklearn import preprocessing
from calculate import *
#from sklearn.metrics.pairwise import cosine_similarity

dir = r'C:\\Users\\ASUS\\Documents\\Programming\\Python\\Algeo02-21067\\dataset\\train'
tesdir = r'C:\\Users\\ASUS\\Documents\\Programming\\Python\\Algeo02-21067\\dataset\\test'

label_training = labelimg(dir)
label_test = labelimg(tesdir)

#Training
A = A_Matrix(dir)
kov = covariance(A)
# kov = np.array([[1,3,0,0],[3,2,1,0],[0,1,3,4],[0,0,4,1]])
# kov = kov.astype('float64')


# #Bagian yang harus kita kulik sendiri, ga boleh pakai library linear algebra
eigenvalues, eigenvectors, = eigenNP(kov)
eigval, eigenvec = theEigen(kov,100)
# print("eigenvector from fucntion: ")
# print(eigenvec)
# print("eigenvector from np:")
# print(eigenvectors)

eigval, eigenvec = eigenSort(eigval,eigenvec)
eigfaces = np.array(get_eigenfaces(eigenvec,50))


a = eigfaces[20].reshape(256,256)
a = np.interp(a, (a.min(), a.max()), (0, 256))
img = (a).reshape(256,256).astype('uint8')

grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.imshow('image',grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# a = np.array([[1, 2], [3, 4]])

# b = np.array([0, 0])

# x = np.linalg.solve(a, b)
# print(x)


