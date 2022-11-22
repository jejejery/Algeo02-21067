# import os
import numpy as np
# import cv2
from time import *
# from PIL import Image
from .pengolahan_citra import *
from .QR_Decomposition import *
from .theEigen import *


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
    eigval, eigenvec = theEigen(kov,60)
    eigval, eigenvec = eigenSort(eigval,eigenvec)
    eigenvec = np.array([k/norm_vector(k) for k in eigenvec]) #normalisasi vektor eigen
    x = int(training.shape[0]*0.75)
    eigfaces = np.array(get_eigenfaces(eigenvec,x,training)) #75% eigenface pertama
    weight_training = get_weight(eigfaces,A)

    return eigfaces, weight_training, avg

def metrics_calculation(test,eigenfaces,weight_training,avg_training,label_training):
    weight_test = np.dot(eigenfaces,test-avg_training)
    ctr = 0
    
    for i in weight_training:
        if ctr == 0:
            euclidian_distance = norm_vector(weight_test-i)
            cos_sim = cosine_sim(weight_test,i)
            mark = 0
        else:
            if norm_vector(weight_test-i) < euclidian_distance:
                euclidian_distance = norm_vector(weight_test-i)
                mark = ctr
                cos_sim = cosine_sim(weight_test,i)
        ctr += 1

    
    return euclidian_distance, cos_sim, label_training[mark]








    

        




    

