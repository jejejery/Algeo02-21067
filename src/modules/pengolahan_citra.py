import os
import cv2
import numpy as np
from PIL import Image
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